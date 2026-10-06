import json
import sqlite3
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsroom.tests.assessor_fixture_support import candidate_fixture
from jsonschema import Draft202012Validator, ValidationError

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes, digest_canonical
from newsroom.control_plane import native_assessor as native_assessor_module
from newsroom.control_plane.admission import DeterministicWriteAdmission
from newsroom.control_plane.evidence import (
    EvidencePackage,
    bounded_named_entities,
    evidence_package_value,
    validate_governed_evidence_records,
)
from newsroom.control_plane.govuk_evidence import parse_govuk_content_document
from newsroom.control_plane.native_assessor import (
    AutonomousNativeEvidenceAssessor,
    CONFIG_IDENTITY,
    CONTEXT_IDENTITY,
    CONTEXT_MANIFEST_SCHEMA_VERSION,
    INPUT_BOUND_VERSION,
    NativeAssessmentExecution,
    NativeAssessmentUsage,
    PROVIDER_SCHEMA,
    PROVIDER_SCHEMA_DIGEST,
    REASSESSABLE_HOLDS,
    SCHEMA,
    SCHEMA_DIGEST,
    SYSTEM,
    VERSION,
    _MAX_RETAINED_RESULT_BYTES,
    assessor_admission_recovery_due,
    native_assessment_input_bound,
)
from newsroom.control_plane.native_evidence import (
    EvidenceAssessor,
    NativeEvidenceController,
    NativeEvidenceError,
    NativeEvidenceHold,
)
from newsroom.control_plane.model_usage import (
    InvocationEfficiencyPolicy,
    ModelUsageIntegrityError,
    ModelUsageService,
    WorkEnvelope,
    WorkloadClass,
)
from newsroom.control_plane.store import connect
from newsroom.increment10.evidence import EvidencePackageError, _base_package
from newsroom.control_plane.writer import (
    CONT_DISABLED_CAPABILITIES,
    CONT_PRIMARY_COMMAND_FLAGS,
    CONT_PRIMARY_MODEL,
    CONT_PRIMARY_REASONING,
    _grok_command_flags,
)
from newsroom.tests.test_increment10_editorial import _ready_package
from newsroom.tests.test_increment10_ingress import _candidate


REVISION = "1" * 40


def _use_historical_v16(monkeypatch):
    """Seed a genuine full-package v16 result; never bypass v18 references."""

    monkeypatch.setattr(
        native_assessor_module, "VERSION", native_assessor_module._V16_PRODUCER_VERSION
    )
    monkeypatch.setattr(native_assessor_module, "SYSTEM", native_assessor_module._V16_SYSTEM)
    monkeypatch.setattr(
        native_assessor_module,
        "PROVIDER_SCHEMA_DIGEST",
        native_assessor_module.SCHEMA_DIGEST,
    )
    monkeypatch.setattr(
        native_assessor_module,
        "CONTEXT_MANIFEST_SCHEMA_VERSION",
        "newsroom.native-evidence-assessor.context-manifest.v2",
    )
    monkeypatch.setattr(
        native_assessor_module, "MODEL", CONT_PRIMARY_MODEL,
    )
    monkeypatch.setattr(
        native_assessor_module, "REASONING", CONT_PRIMARY_REASONING,
    )
    monkeypatch.setattr(
        native_assessor_module, "COMMAND_FLAGS", CONT_PRIMARY_COMMAND_FLAGS,
    )


def _empty_reference_result(reason="No supported new information."):
    return {
        "package": {
            "select_new_information": False,
            "governed_claims": [],
            "qualification_evidence": [],
            "selection_rationale": reason,
            "geography": [],
            "categories": [],
            "explicit_exclusions": [],
        }
    }


def _v18_wire_from_v17(value):
    """Convert test fixtures only; production accepts no v17-shaped v18 output."""

    from newsroom.control_plane.admission import _QUALIFICATION_CLASSIFIER_FIELDS

    package = value["package"]
    claims = []
    for item in package["governed_claims"]:
        claims.append({
            "claim_role": item["claim_role"],
            "status": item["status"],
            "source_range": item["claim_range"],
            "rendered_assertion_zh_hant_hk_fragments": item[
                "rendered_fragments"
            ],
            "factual_localisations": [
                {
                    "source_lookup_key": source,
                    "rendered_expression": rendered,
                }
                for source, rendered in item["localised_factual_expressions"]
            ],
            "quotation_source_keys": item["quotations"],
        })
    qualifications = []
    for item in package["qualification_evidence"]:
        qualifications.append({
            "test": item["test"],
            "claim_index": item["claim_index"],
            "test_evidence": {
                (
                    key
                    if key in _QUALIFICATION_CLASSIFIER_FIELDS
                    else key + "_source_lookup_key"
                ): witness
                for key, witness in item["test_evidence"].items()
            },
        })
    return {"package": {
        **{
            key: package[key]
            for key in (
                "substantive_claim_indexes",
                "selection_rationale",
                "geography",
                "categories",
                "explicit_exclusions",
            )
        },
        "governed_claims": claims,
        "qualification_evidence": qualifications,
    }}


def _current_wire_from_v17(value):
    wire = _v18_wire_from_v17(value)
    wire["package"]["select_new_information"] = bool(
        wire["package"].pop("substantive_claim_indexes")
    )
    return wire


@pytest.mark.parametrize(("changes", "expected"), [
    ({}, True),
    ({"reason": "ASSESSOR_TRANSPORT_HOLD"}, False),
    ({"failure_class": "OtherError"}, False),
    ({"assessment_started_at": "2026-09-27T10:00:00+00:00"}, False),
    ({"assessment_started_at": None}, True),
])
def test_assessor_admission_recovery_predicate_is_pre_dispatch_only(changes, expected):
    facts = {
        "reason": "ACQUISITION_RESULT_NOT_RETAINED",
        "failure_class": "ModelUsageAdmissionError",
    }
    assert assessor_admission_recovery_due(facts | changes) is expected


def _model_package_value(package):
    value = evidence_package_value(package)
    return {
        "substantive_new_information": value["substantive_new_information"],
        "governed_claims": [
            {
                key: item[key]
                for key in (
                    "claim_id", "claim", "passage_index", "supporting_excerpt",
                    "source_ids", "status",
                    "rendered_assertion_zh_hant_hk", "claim_role",
                    "localised_factual_expressions", "quotations", "certainty",
                    "originality_basis", "originality_policy_version",
                    "admitted_use", "policy_version",
                )
            } | {
                "semantic_relation": {
                    "source_modality": "ASSERTED",
                    "rendered_modality": "ASSERTED",
                    "source_polarity": "AFFIRMED",
                    "rendered_polarity": "AFFIRMED",
                    "relation": "SEMANTICALLY_EQUIVALENT",
                }
            }
            for item in value["governed_claims"]
        ],
        "qualification_evidence": [
            {
                key: item[key]
                for key in (
                    "test", "governed_claim_id", "policy_version"
                )
            } | {"test_evidence": dict(item["test_evidence"])}
            for item in value["qualification_evidence"]
        ],
        "selection_rationale": value["selection_rationale"],
        "geography": value["geography"],
        "categories": value["categories"],
        "explicit_exclusions": value["explicit_exclusions"],
    }


def test_native_assessor_schema_is_closed_and_accepts_the_exact_package_shape(tmp_path) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    assessed = _ready_package(candidate)[1]
    validator = Draft202012Validator(SCHEMA)
    validator.validate({"package": _model_package_value(base)})
    model_value = _model_package_value(assessed)
    validator.validate({"package": model_value})
    assert "source_authority_decision_ids" not in model_value["governed_claims"][0]
    assert "semantic_relation_evidence_id" not in model_value["governed_claims"][0]
    assert "qualification_record_id" not in model_value["qualification_evidence"][0]
    named = json.loads(canonical_json_bytes({"package": model_value}))
    named["package"]["governed_claims"][0]["named_entities"] = []
    with pytest.raises(ValidationError):
        validator.validate(named)
    invalid_qualification = json.loads(canonical_json_bytes({"package": model_value}))
    invalid_qualification["package"]["qualification_evidence"][0][
        "test_evidence"
    ]["invented"] = "value"
    with pytest.raises(ValidationError):
        validator.validate(invalid_qualification)
    invalid = _model_package_value(base)
    invalid["invented"] = True
    with pytest.raises(ValidationError):
        validator.validate({"package": invalid})
    invalid_semantic = _model_package_value(assessed)
    invalid_semantic["governed_claims"][0]["semantic_relation"].update({
        "source_modality": "ALLOWS",
        "rendered_modality": "ALLOWS",
        "relation": "EQUIVALENT",
    })
    with pytest.raises(ValidationError):
        validator.validate({"package": invalid_semantic})
    invalid_category = _model_package_value(assessed)
    invalid_category["categories"] = ["immigration"]
    with pytest.raises(ValidationError):
        validator.validate({"package": invalid_category})
    invalid_geography = _model_package_value(assessed)
    invalid_geography["geography"] = ["Britain"]
    with pytest.raises(ValidationError):
        validator.validate({"package": invalid_geography})
    assert VERSION == "newsroom.native-evidence-assessor.v23"
    assert "ASSESSOR_CLAIM_BINDING_HOLD" in REASSESSABLE_HOLDS
    legacy = native_assessor_module._V17_SYSTEM
    assert "whitespace, newlines and country labels exactly" in legacy
    assert "unfamiliar official source-bound literal" in legacy
    assert "calendar months as months without converting them" in legacy
    assert "calendar years as years without converting them" in legacy
    assert "Ordinary unit and process nouns must be translated" in legacy
    assert "excerpt-only entities" not in legacy
    assert "first observation of an old clause" in legacy
    assert "DELETED" in legacy
    assert "Only rendered_assertion_zh_hant_hk is translated" in legacy
    assert "must exactly equal the named-entity set in claim" in legacy
    assert "only in the excerpt, source body or inventory" in legacy
    assert "must not be copied unchanged" in legacy
    assert "byte-for-byte from the claim or supporting excerpt" in legacy
    assert "Never paraphrase or invent new_state" in legacy
    assert "rendered_assertion_zh_hant_hk_fragments" in SYSTEM
    assert "one contiguous source_range" in SYSTEM
    assert "ending _source_lookup_key" in SYSTEM
    assert "substantive_claim_indexes" not in SYSTEM
    assert "select_new_information false" in SYSTEM
    connection.close()


@pytest.mark.parametrize(
    ("claim_text", "excerpt", "rendered", "expected_entities"),
    (
        # Retained result 27257: rejecting "The Department" must not consume
        # the start of the exact institution name which follows the article.
        (
            "The Department for Education will work with the Food Standards Agency on an approach to monitor school food",
            "The Department for Education will work with the Food Standards Agency on an approach to monitor school food.",
            "Department for Education 將與 Food Standards Agency 合作制訂監察學校膳食的方法。",
            ("Department for Education", "Food Standards Agency"),
        ),
        (
            "The Home Office published changes",
            "The Home Office published changes to the Skilled Worker Visa.",
            "Home Office 已公布修訂。",
            ("Home Office",),
        ),
        (
            "The University of Salford published guidance",
            "The University of Salford published guidance.",
            "University of Salford公布指引。",
            ("University of Salford",),
        ),
        # Retained source-bound rendering failure, ledger result 25858.
        (
            "Responsibility for the overall apprenticeship programme now sits with the Department of Work and Pensions (DWP).",
            "Responsibility for the overall apprenticeship programme now sits with the Department of Work and Pensions (DWP).",
            "整體學徒計劃的責任現時由 Department of Work and Pensions（DWP）承擔。",
            ("DWP", "Department of Work and Pensions"),
        ),
        (
            "During this period, the EPA version of the apprenticeship will remain available until the new apprenticeship assessment version is formally released for starts.",
            "During this period, the EPA version of the apprenticeship will remain available until the new apprenticeship assessment version is formally released for starts.",
            "在此期間，該學徒計劃的 EPA 版本會繼續可供取用，直至新學徒評核版本正式開放予開辦為止。",
            ("EPA",),
        ),
    ),
)
def test_native_assessor_derives_entities_from_constructed_uk03_output(
    tmp_path, claim_text, excerpt, rendered, expected_entities,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    package = _model_package_value(_ready_package(candidate)[1])
    claim = package["governed_claims"][0]
    claim.update({
        "claim": claim_text,
        "supporting_excerpt": excerpt,
        "rendered_assertion_zh_hant_hk": rendered,
    })
    package.update({
        "substantive_new_information": [claim_text],
        "governed_claims": [claim],
        "qualification_evidence": [],
    })
    role = SimpleNamespace(
        role=SimpleNamespace(value="ORIGINATING_AUTHORITY"),
        purpose="Own immigration rules",
        canonical_value=lambda: {
            "role": "ORIGINATING_AUTHORITY", "purpose": "Own immigration rules",
        },
    )
    source = SimpleNamespace(
        unit=SimpleNamespace(
            source_id="source-1",
            authority=SimpleNamespace(definition_id="definition-1"),
        ),
        source_version=SimpleNamespace(
            canonical_digest="sha256:" + "a" * 64,
            request=SimpleNamespace(roles=(role,)),
        ),
        rights=SimpleNamespace(
            record_id="rights-1",
            decision="PERMITTED",
            permitted_use="PUBLICATION_EVIDENCE",
        ),
        dependency=SimpleNamespace(
            record_id="dependency-1",
            dependency_status="RESOLVED",
            evidential_origin_id="origin-1",
            originating_report_id="report-1",
        ),
    )
    body = excerpt.encode()
    acquired = SimpleNamespace(
        currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
        body_origin="",
        receipt_digest="sha256:" + "b" * 64,
        canonical_url="https://www.gov.uk/example",
        publisher="Home Office",
        responsible_body="Home Office",
        source_type="PRIMARY_OFFICIAL",
        publication_time="2026-09-09T12:00:00.000000Z",
        retrieval_time="2026-09-09T12:01:00.000000Z",
        source_updated_time="2026-09-09T12:00:00.000000Z",
        transport_evidence_digest="sha256:" + "c" * 64,
        geography="UK",
        language="en-GB",
        body=body,
        body_digest=digest_bytes(body),
    )

    invalid_semantic = json.loads(canonical_json_bytes({"package": package}))
    invalid_semantic["package"]["governed_claims"][0][
        "semantic_relation"
    ].update({
        "source_modality": "ALLOWS",
        "rendered_modality": "ALLOWS",
        "relation": "EQUIVALENT",
    })
    with pytest.raises(EvidencePackageError, match="semantic relation"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(
                canonical_json_bytes(invalid_semantic).decode(), {}
            ),
            candidate,
            base,
            (source,),
            (acquired,),
        )

    result = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(
            canonical_json_bytes({"package": package}).decode(), {}
        ),
        candidate,
        base,
        (source,),
        (acquired,),
    )

    assert result.governed_claims[0].named_entities == expected_entities
    assert result.governed_claims[0].rendered_named_entities == expected_entities
    governed = replace(
        base,
        substantive_new_information=result.substantive_new_information,
        governed_claims=result.governed_claims,
        qualification_evidence=result.qualification_evidence,
        selection_rationale=result.selection_rationale,
        geography=result.geography,
        categories=result.categories,
        explicit_exclusions=result.explicit_exclusions,
    )
    records = NativeEvidenceController._records(
        base, governed, (source,), (acquired,), result
    )
    retained_rows = tuple(
        (
            record["record_id"],
            record["record_type"],
            canonical_json_bytes(record).decode(),
            digest_bytes(canonical_json_bytes(record)),
        )
        for record in records
    )
    assert validate_governed_evidence_records(
        candidate_id=candidate.candidate_id,
        source_inventory=(("source-1", acquired.canonical_url),),
        base_package_digest=base.digest,
        package=governed,
        retained_records=retained_rows,
    ) is not None
    def decide(assessment, passage, information):
        admitted_package = replace(
            _ready_package(candidate)[1],
            passages=(passage,),
            substantive_new_information=(information,),
            governed_claims=assessment.governed_claims,
            qualification_evidence=(),
            resolved_evidence_records=tuple(
                (
                    record["record_id"],
                    digest_bytes(canonical_json_bytes(record)),
                )
                for record in assessment.assessment_records
            ),
        )
        return DeterministicWriteAdmission().decide_candidate_identity(
            candidate_id=admitted_package.candidate_id,
            hypothesis_id=admitted_package.hypothesis_id,
            package=admitted_package,
            decided_at="2026-09-09T12:02:00.000000Z",
        )

    decision = decide(result, excerpt, claim_text)
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decision.stable_reason_codes
    if expected_entities != ("Home Office",):
        connection.close()
        return

    ancestry_claim = (
        "English language requirement for settlement on the UK Ancestry route UKA "
        "15.1. Unless an exemption applies, the applicant must: (a) Where the date "
        "of application is before 26 March 2027, the applicant must, unless an "
        "exemption applies, show English language ability on the Common European "
        "Framework of Reference for Languages in speaking and listening to at least "
        "level B1; or (b) Where the date of application is on or after 26 March 2027, "
        "the applicant must, unless an exemption applies, show English language "
        "ability on the Common European Framework of Reference for Languages in "
        "speaking and listening to at least level B2."
    )
    ancestry = json.loads(canonical_json_bytes(package))
    ancestry["governed_claims"][0].update({
        "claim": ancestry_claim,
        "supporting_excerpt": ancestry_claim,
        "rendered_assertion_zh_hant_hk": (
            "UK Ancestry定居途徑的英語要求：如申請日期早於2027年3月26日，"
            "除非獲豁免，申請人的聆聽及口語能力須達Common European Framework "
            "of Reference for Languages至少B1級；如申請日期為2027年3月26日或"
            "之後，則須至少達B2級。"
        ),
        "localised_factual_expressions": [
            ["26 March 2027", "2027年3月26日"],
        ],
    })
    ancestry["substantive_new_information"] = [ancestry_claim]
    ancestry_acquired = SimpleNamespace(**{
        **vars(acquired), "body": ancestry_claim.encode(),
    })
    ancestry_assessment = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(
            canonical_json_bytes({"package": ancestry}).decode(), {}
        ),
        candidate, base, (source,), (ancestry_acquired,),
    )
    assert ancestry_assessment.governed_claims[0].named_entities == (
        "B1", "B2", "Common European Framework of Reference for Languages",
        "UK Ancestry",
    )
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decide(
        ancestry_assessment, ancestry_claim, ancestry_claim
    ).stable_reason_codes
    ancestry_near_match = json.loads(canonical_json_bytes(ancestry))
    ancestry_near_match["governed_claims"][0][
        "rendered_assertion_zh_hant_hk"
    ] = ancestry_near_match["governed_claims"][0][
        "rendered_assertion_zh_hant_hk"
    ].replace("B2級", "B2X級")
    with pytest.raises(EvidencePackageError, match="rendered named entities differ"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(
                canonical_json_bytes({"package": ancestry_near_match}).decode(), {}
            ),
            candidate, base, (source,), (ancestry_acquired,),
        )

    for exact_claim, rendered, names in (
        (
            "The applicant must be in the UK.", "申請人必須身在UK。",
            ("UK",),
        ),
        (
            "John Smith said services would resume.",
            "John Smith表示服務將恢復。", ("John Smith",),
        ),
        (
            "Appendix Victim of Domestic Abuse applies.",
            "適用Appendix Victim of Domestic Abuse。",
            ("Appendix Victim of Domestic Abuse",),
        ),
        (
            "General Grounds for Refusal applies.",
            "適用General Grounds for Refusal。",
            ("General Grounds for Refusal",),
        ),
        (
            "AR(EU)1.1 applies.", "適用AR(EU)1.1。", ("AR(EU)1.1",),
        ),
        (
            "Appendix O applies.", "適用Appendix O。", ("Appendix O",),
        ),
        # Reproduce the omitted source-bound ETA token, not a complete live result.
        (
            "Applicants travelling to the UK must obtain an ETA.",
            "前往UK的申請人必須取得ETA。",
            ("ETA", "UK"),
        ),
        (
            "This route is for ECAA workers, business persons and their family "
            "members who are in the UK and already hold permission in that capacity "
            "and are seeking an extension of their permission.",
            "ECAA工作者、商務人士及其家屬如身在UK並已持有相關許可，"
            "可申請延長許可。",
            ("ECAA", "UK"),
        ),
    ):
        current = json.loads(canonical_json_bytes(package))
        current["governed_claims"][0].update({
            "claim": exact_claim, "supporting_excerpt": exact_claim,
            "rendered_assertion_zh_hant_hk": rendered,
        })
        current["substantive_new_information"] = [exact_claim]
        current_acquired = SimpleNamespace(**{
            **vars(acquired), "body": exact_claim.encode(),
        })
        assessment = AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(canonical_json_bytes({"package": current}).decode(), {}),
            candidate, base, (source,), (current_acquired,),
        )
        assert assessment.governed_claims[0].named_entities == names
        assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decide(
            assessment, exact_claim, exact_claim
        ).stable_reason_codes

        current["governed_claims"][0]["rendered_assertion_zh_hant_hk"] += " unsupported prose"
        with pytest.raises(NativeEvidenceHold, match="ASSESSOR_RENDERING_CONTRACT_HOLD"):
            AutonomousNativeEvidenceAssessor._validated_execution(
                NativeAssessmentExecution(canonical_json_bytes({"package": current}).decode(), {}),
                candidate, base, (source,), (current_acquired,),
            )

    combined_claim = (
        "From 11 November 2025 all references to General Grounds for Refusal are "
        "to be read as Part Suitability."
    )
    combined = json.loads(canonical_json_bytes(package))
    combined["governed_claims"][0].update({
        "claim": combined_claim,
        "supporting_excerpt": combined_claim,
        "rendered_assertion_zh_hant_hk": (
            "由2025年11月11日起，所有對General Grounds for Refusal的提述須"
            "理解為Part Suitability。"
        ),
        "localised_factual_expressions": [
            ["11 November 2025", "2025年11月11日"]
        ],
    })
    combined["substantive_new_information"] = [combined_claim]
    combined_acquired = SimpleNamespace(**{
        **vars(acquired), "body": combined_claim.encode(),
    })
    combined_assessment = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(
            canonical_json_bytes({"package": combined}).decode(), {}
        ),
        candidate, base, (source,), (combined_acquired,),
    )
    assert combined_assessment.governed_claims[0].named_entities == (
        "General Grounds for Refusal", "Part Suitability",
    )
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decide(
        combined_assessment, combined_claim, combined_claim
    ).stable_reason_codes

    for exact_claim, source_context, rendered, names in (
        (
            "The relevant qualification meets ST8.1/2/3 of the Rules.",
            "Immigration Rules Appendix Graduate. The relevant qualification "
            "meets ST8.1/2/3 of the Rules.",
            "有關資格符合ST8.1/2/3規則。",
            ("ST8.1/2/3",),
        ),
        (
            "Permission was previously granted under Part 14: stateless persons.",
            "Immigration Rules part 14: stateless persons. Permission was "
            "previously granted under Part 14: stateless persons.",
            "先前已根據Part 14: stateless persons獲准。",
            ("Part 14: stateless persons",),
        ),
        (
            "A Stateless person or their partner or dependent child previously "
            "granted permission under Part 14: stateless persons applying on or "
            "after 11 November 2025 will be considered under this route.",
            "Immigration Rules part 14: stateless persons. A Stateless person or "
            "their partner or dependent child previously granted permission under "
            "Part 14: stateless persons applying on or after 11 November 2025 will "
            "be considered under this route.",
            "先前根據 Part 14: stateless persons 獲准逗留的無國籍人士或其伴侶或"
            "受養子女，如在2025年11月11日或之後提出申請，將按此途徑審理。",
            ("Part 14: stateless persons",),
        ),
    ):
        official = json.loads(canonical_json_bytes(package))
        official["governed_claims"][0].update({
            "claim": exact_claim,
            "supporting_excerpt": exact_claim,
            "rendered_assertion_zh_hant_hk": rendered,
        })
        official["substantive_new_information"] = [exact_claim]
        official_acquired = SimpleNamespace(**{
            **vars(acquired), "body": source_context.encode(),
        })
        official_assessment = AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(
                canonical_json_bytes({"package": official}).decode(), {}
            ),
            candidate, base, (source,), (official_acquired,),
        )
        assert official_assessment.governed_claims[0].named_entities == names
        assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decide(
            official_assessment, source_context, exact_claim
        ).stable_reason_codes

    month_claim = (
        "If the applicant meets the ECAA business person requirement, they will be "
        "granted permission to stay for up to 36 months."
    )
    month = json.loads(canonical_json_bytes(package))
    month["governed_claims"][0].update({
        "claim": month_claim,
        "supporting_excerpt": month_claim,
        "rendered_assertion_zh_hant_hk": (
            "申請人如符合ECAA商務人士要求，可獲准逗留最多36個月。"
        ),
        "localised_factual_expressions": [["36 months", "36個月"]],
    })
    month["substantive_new_information"] = [month_claim]
    month_acquired = SimpleNamespace(**{
        **vars(acquired), "body": month_claim.encode(),
    })
    month_assessment = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(
            canonical_json_bytes({"package": month}).decode(), {}
        ),
        candidate, base, (source,), (month_acquired,),
    )
    assert month_assessment.governed_claims[0].localised_factual_expressions == (
        ("36 months", "36個月"),
    )
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decide(
        month_assessment, month_claim, month_claim
    ).stable_reason_codes

    year_claim = "The applicant will be granted permission for 5 years."
    year = json.loads(canonical_json_bytes(package))
    year["governed_claims"][0].update({
        "claim": year_claim,
        "supporting_excerpt": year_claim,
        "rendered_assertion_zh_hant_hk": "申請人會獲批為期5年的許可。",
        "localised_factual_expressions": [["5 years", "5年"]],
    })
    year["substantive_new_information"] = [year_claim]
    year_acquired = SimpleNamespace(**{
        **vars(acquired), "body": year_claim.encode(),
    })
    year_assessment = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(
            canonical_json_bytes({"package": year}).decode(), {}
        ),
        candidate, base, (source,), (year_acquired,),
    )
    assert year_assessment.governed_claims[0].localised_factual_expressions == (
        ("5 years", "5年"),
    )
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decide(
        year_assessment, year_claim, year_claim
    ).stable_reason_codes

    boundary_claim = "Changes were published by the Home Office"
    boundary_excerpt = "Home Office announced changes."
    boundary_body = f"{boundary_claim}. {boundary_excerpt}"
    boundary_package = _model_package_value(_ready_package(candidate)[1])
    boundary_package["governed_claims"][0].update({
        "claim": boundary_claim,
        "supporting_excerpt": boundary_excerpt,
        "rendered_assertion_zh_hant_hk": "Home Office 已公布有關修訂。",
    })
    boundary_package.update({
        "substantive_new_information": [boundary_claim],
        "governed_claims": [boundary_package["governed_claims"][0]],
        "qualification_evidence": [],
    })
    boundary_acquired = SimpleNamespace(**{
        **vars(acquired), "body": boundary_body.encode(),
    })
    assert bounded_named_entities(f"{boundary_claim}\n{boundary_excerpt}") != (
        bounded_named_entities(boundary_claim)
        | bounded_named_entities(boundary_excerpt)
    )
    boundary_result = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(
            canonical_json_bytes({"package": boundary_package}).decode(), {}
        ),
        candidate, base, (source,), (boundary_acquired,),
    )
    boundary_decision = decide(boundary_result, boundary_body, boundary_claim)
    assert (
        "INVALID_GOVERNED_CLAIM_EVIDENCE"
        not in boundary_decision.stable_reason_codes
    )

    rule_claim = (
        "An applicant may not apply for an administrative review of an eligible "
        "decision, as defined in paragraph AR(EU)1.1., where that decision was "
        "made on or after 5 October 2023."
    )
    rule_excerpt = "AR(EU)1.4. " + rule_claim
    rule_package = json.loads(canonical_json_bytes(package))
    rule_package["governed_claims"][0].update({
        "claim": rule_claim,
        "supporting_excerpt": rule_excerpt,
        "rendered_assertion_zh_hant_hk": (
            "申請人不得就AR(EU)1.1所界定、於2023年10月5日或之後作出的"
            "合資格決定申請行政覆核。"
        ),
        "localised_factual_expressions": [
            ["5 October 2023", "2023年10月5日"]
        ],
    })
    rule_package["substantive_new_information"] = [rule_claim]
    rule_acquired = SimpleNamespace(**{
        **vars(acquired), "body": rule_excerpt.encode(),
    })
    rule_assessment = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(
            canonical_json_bytes({"package": rule_package}).decode(), {}
        ),
        candidate, base, (source,), (rule_acquired,),
    )
    assert rule_assessment.governed_claims[0].named_entities == ("AR(EU)1.1",)
    assert "INVALID_GOVERNED_CLAIM_EVIDENCE" not in decide(
        rule_assessment, rule_excerpt, rule_claim
    ).stable_reason_codes

    invented_excerpt_entity = json.loads(canonical_json_bytes({
        "package": rule_package
    }))
    invented_excerpt_entity["package"]["governed_claims"][0][
        "rendered_assertion_zh_hant_hk"
    ] += " AR(EU)1.4"
    with pytest.raises(EvidencePackageError, match="rendered named entities"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(
                canonical_json_bytes(invented_excerpt_entity).decode(), {}
            ),
            candidate, base, (source,), (rule_acquired,),
        )

    missing_claim_entity = json.loads(canonical_json_bytes({
        "package": rule_package
    }))
    missing_claim_entity["package"]["governed_claims"][0][
        "supporting_excerpt"
    ] = rule_excerpt.replace("AR(EU)1.1", "AR(EU)1.2")
    missing_acquired = SimpleNamespace(**{
        **vars(acquired),
        "body": (
            rule_claim + " "
            + missing_claim_entity["package"]["governed_claims"][0][
                "supporting_excerpt"
            ]
        ).encode(),
    })
    with pytest.raises(EvidencePackageError, match="source evidence"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(
                canonical_json_bytes(missing_claim_entity).decode(), {}
            ),
            candidate, base, (source,), (missing_acquired,),
        )
    paraphrased = json.loads(canonical_json_bytes({"package": package}))
    paraphrased["package"]["governed_claims"][0]["claim"] = (
        "The Home Office changed the immigration system"
    )
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_CLAIM_BINDING_HOLD"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(
                canonical_json_bytes(paraphrased).decode(), {}
            ),
            candidate, base, (source,), (acquired,),
        )
    unsupported = json.loads(canonical_json_bytes({"package": package}))
    unsupported["package"]["governed_claims"][0][
        "supporting_excerpt"
    ] = "published changes to the Skilled Worker Visa."
    with pytest.raises(EvidencePackageError, match="source evidence"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(canonical_json_bytes(unsupported).decode(), {}),
            candidate, base, (source,), (acquired,),
        )
    changed = json.loads(canonical_json_bytes({"package": package}))
    changed["package"]["governed_claims"][0][
        "rendered_assertion_zh_hant_hk"
    ] = "已公布修訂。"
    with pytest.raises(EvidencePackageError, match="rendered named entities"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(canonical_json_bytes(changed).decode(), {}),
            candidate, base, (source,), (acquired,),
        )
    invented = json.loads(canonical_json_bytes({"package": package}))
    invented["package"]["governed_claims"][0][
        "rendered_assertion_zh_hant_hk"
    ] += " NHS England"
    with pytest.raises(EvidencePackageError, match="rendered named entities"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(canonical_json_bytes(invented).decode(), {}),
            candidate, base, (source,), (acquired,),
        )

    for exact_claim, rendered, error, message in (
        (
            "All other qualifications 18 months if the applicant has successfully "
            "completed a course at UK bachelor’s degree or master’s degree level, or "
            "another relevant qualification that meets the requirement of ST8.1/2/3 "
            "of the Rules, and the application is made on or after 1 January 2027.",
            "其他所有學歷如申請人已成功完成英國學士或碩士學位程度課程，則獲准"
            "逗留18個月。",
            EvidencePackageError,
            "rendered named entities",
        ),
        (
            "1232 Residential, day and domiciliary care managers and proprietors – "
            "all jobs Yes Yes Yes Yes 31 December 2026",
            "職業代碼1232的所有職位列入Appendix Immigration Salary List，剔除"
            "日期為2026年12月31日。",
            NativeEvidenceHold,
            "ASSESSOR_RENDERING_CONTRACT_HOLD",
        ),
        (
            "An occupation is only included on the list where an application has "
            "been made using a certificate of sponsorship issued by a sponsor to an "
            "applicant before the removal date stated in the table.",
            "An occupation is only included on the list where an application has "
            "been made using a certificate of sponsorship issued by a sponsor to an "
            "applicant before the removal date stated in the table.",
            EvidencePackageError,
            "package value is invalid",
        ),
    ):
        retained_failure = json.loads(canonical_json_bytes({"package": package}))
        retained_claim = retained_failure["package"]["governed_claims"][0]
        retained_claim.update({
            "claim": exact_claim,
            "supporting_excerpt": exact_claim,
            "rendered_assertion_zh_hant_hk": rendered,
        })
        retained_failure["package"]["substantive_new_information"] = [exact_claim]
        retained_acquired = SimpleNamespace(**{
            **vars(acquired), "body": exact_claim.encode(),
        })
        with pytest.raises(error, match=message):
            AutonomousNativeEvidenceAssessor._validated_execution(
                NativeAssessmentExecution(
                    canonical_json_bytes(retained_failure).decode(), {}
                ),
                candidate, base, (source,), (retained_acquired,),
            )

    source_claim = "T2 Minister of Religion is a route to settlement."
    translated_claim = "T2 Minister of Religion是通往settlement的途徑。"
    retained_binding_failure = json.loads(canonical_json_bytes({"package": package}))
    retained_binding_claim = retained_binding_failure["package"]["governed_claims"][0]
    retained_binding_claim.update({
        "claim": translated_claim,
        "supporting_excerpt": source_claim,
        "rendered_assertion_zh_hant_hk": translated_claim,
    })
    retained_binding_failure["package"]["substantive_new_information"] = [
        translated_claim
    ]
    retained_binding_acquired = SimpleNamespace(**{
        **vars(acquired), "body": source_claim.encode(),
    })
    with pytest.raises(EvidencePackageError, match="package value is invalid"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            NativeAssessmentExecution(
                canonical_json_bytes(retained_binding_failure).decode(), {}
            ),
            candidate, base, (source,), (retained_binding_acquired,),
        )

    for source_claim, altered_span in (
        (
            "Immigration Rules part 4: work experience\n\n“Au pair” placements "
            "DELETED Working holidaymakers DELETED",
            "Immigration Rules part 4: work experience “Au pair” placements "
            "DELETED Working holidaymakers DELETED",
        ),
        (
            "for travel to the UK on or after 8 January 2025: Antigua and Barbuda "
            "Argentina Australia Barbados Belize Brazil Brunei Canada Chile Costa "
            "Rica Grenada Guatemala Guyana Hong Kong Special Administrative Region",
            "for travel to the UK on or after 8 January 2025: Hong Kong Special "
            "Administrative Region",
        ),
    ):
        for field in ("claim", "supporting_excerpt"):
            altered = json.loads(canonical_json_bytes({"package": package}))
            altered_claim = altered["package"]["governed_claims"][0]
            altered_claim.update({
                "claim": source_claim,
                "supporting_excerpt": source_claim,
                "rendered_assertion_zh_hant_hk": (
                    "UK及Hong Kong內容。" if "Hong Kong" in source_claim else "內容。"
                ),
            })
            altered_claim[field] = altered_span
            altered["package"]["substantive_new_information"] = [
                altered_claim["claim"]
            ]
            altered_acquired = SimpleNamespace(**{
                **vars(acquired), "body": source_claim.encode(),
            })
            with pytest.raises(
                NativeEvidenceHold, match="ASSESSOR_CLAIM_BINDING_HOLD"
            ):
                AutonomousNativeEvidenceAssessor._validated_execution(
                    NativeAssessmentExecution(
                        canonical_json_bytes(altered).decode(), {}
                    ),
                    candidate, base, (source,), (altered_acquired,),
                )
    connection.close()


@pytest.fixture
def retained_22589_assessment():
    fixture_root = Path(__file__).parent / "fixtures/native_assessor"
    raw_source = (fixture_root / "uk03-appendix-statelessness.json").read_bytes()
    execution_text = (fixture_root / "result-22589.json").read_text()
    document = parse_govuk_content_document(
        "https://www.gov.uk/guidance/immigration-rules/"
        "immigration-rules-appendix-statelessness",
        raw_source,
        retrieved_at=datetime(2026, 9, 13, tzinfo=UTC),
    )
    passage = document.title + "\n\n" + document.body_text
    body = passage.encode()
    base = EvidencePackage(
        candidate_id="288f61b7-aad5-40c8-a9d7-ef0cd8785bbd",
        hypothesis_id="cc71a3a8-ad84-55c4-8ef0-3352a9abd49e",
        signal_ids=("e0ed0425-a963-4168-831b-e3b71c15c00b",),
        lead_ids=("11bd4906-c706-43b4-be5f-0b5d2e67d607",),
        source_ids=("UK-03",),
        observation_digests=(digest_bytes(body),),
        passages=(passage,),
    )
    assert digest_bytes(raw_source) == (
        "sha256:c6fca2914d4a72085398561cb7b2883b63ae436a8eb6943b3634b34bf57daa2b"
    )
    assert base.digest == (
        "sha256:5ad91778634c32c631875f2559b55f7478f97c13e5e9d031c88f03a2800249b3"
    )
    role = SimpleNamespace(
        role=SimpleNamespace(value="ORIGINATING_AUTHORITY"),
        purpose="Own immigration rules",
        canonical_value=lambda: {
            "role": "ORIGINATING_AUTHORITY", "purpose": "Own immigration rules",
        },
    )
    source = SimpleNamespace(
        unit=SimpleNamespace(
            source_id="UK-03",
            authority=SimpleNamespace(definition_id="definition-uk03"),
        ),
        source_version=SimpleNamespace(
            canonical_digest="sha256:" + "a" * 64,
            request=SimpleNamespace(roles=(role,)),
        ),
        rights=SimpleNamespace(
            record_id="rights-uk03", decision="PERMITTED",
            permitted_use="PUBLICATION_EVIDENCE",
        ),
        dependency=SimpleNamespace(
            record_id="dependency-uk03", dependency_status="RESOLVED",
            evidential_origin_id="origin-uk03", originating_report_id="report-uk03",
        ),
    )
    acquired = SimpleNamespace(
        currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
        body_origin="",
        receipt_digest="sha256:" + "b" * 64,
        canonical_url=(
            "https://www.gov.uk/guidance/immigration-rules/"
            "immigration-rules-appendix-statelessness"
        ),
        publisher="Home Office", responsible_body="Home Office",
        source_type="PRIMARY_OFFICIAL",
        publication_time="2016-02-25T09:18:56.000000Z",
        retrieval_time="2026-09-13T04:25:26.000000Z",
        source_updated_time="2026-08-03T10:12:09.000000Z",
        transport_evidence_digest="sha256:" + "c" * 64,
        geography="UK", language="en-GB", body=body,
        body_digest=digest_bytes(body),
    )

    return base, source, acquired, NativeAssessmentExecution(execution_text, {})


def test_retained_22589_qualification_matches_current_admission_contract(
    retained_22589_assessment,
) -> None:
    from newsroom.control_plane.admission import _qualification_relation_is_proven
    from newsroom.control_plane.evidence import QualificationEvidence

    base, source, acquired, execution = retained_22589_assessment
    candidate = SimpleNamespace(candidate_id=base.candidate_id)
    raw = json.loads(execution.text)
    qualifications = raw["package"].pop("qualification_evidence")
    raw["package"]["qualification_evidence"] = []
    # Preserve the independent retained inline-reference/rendering regression.
    rendered = AutonomousNativeEvidenceAssessor._validated_execution(
        NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {}),
        candidate, base, (source,), (acquired,),
    )
    assert tuple(claim.named_entities for claim in rendered.governed_claims) == (
        ("Part 14: stateless persons",), (),
    )
    by_id = {claim.claim_id: claim for claim in rendered.governed_claims}
    assert any(not _qualification_relation_is_proven(
        QualificationEvidence(
            item["test"], item["governed_claim_id"], "fixture-qualification",
            tuple(item["test_evidence"].items()), item["policy_version"],
        ), by_id[item["governed_claim_id"]], source_context=acquired.body.decode("utf-8"),
    ) for item in qualifications)
    # This asserts producer/consumer parity, not a new interpretation of the
    # retained policy language: the admission predicate remains unchanged.
    with pytest.raises(EvidencePackageError, match="qualification"):
        AutonomousNativeEvidenceAssessor._validated_execution(
            execution, candidate, base, (source,), (acquired,),
        )


def _qualification_assessor_inputs(retained_22589_assessment, *, kind="deadline"):
    from newsroom.control_plane.native_evidence import (
        PublicationRightsAssessment, rights_eligibility_digest,
    )

    base, source, acquired, retained = retained_22589_assessment
    raw = json.loads(retained.text)
    body = acquired.body
    if kind != "retained":
        claim_text, rendering = {
            "deadline": ("The deadline changed.", "限期已經更改。"),
            "policy": ("The policy changed.", "政策已經更改。"),
            "disruption": ("The service was suspended for 90 minutes.", "服務暫停90分鐘。"),
        }[kind]
        claim = raw["package"]["governed_claims"][0]
        claim.update({
            "claim": claim_text, "supporting_excerpt": claim_text,
            "rendered_assertion_zh_hant_hk": rendering,
            "localised_factual_expressions": [],
        })
        test, witnesses = {
            "deadline": ("OFFICIAL_ACTION_OR_DEADLINE", {
                "action_class": "OFFICIAL_DEADLINE", "event_polarity": "AFFIRMED",
                "action_relation": "NEW_OR_CHANGED_OFFICIAL_ACTION",
                "material_relation_span": claim_text, "reader_action": claim_text,
            }),
            "policy": ("LAW_RIGHT_STATUS_POLICY", {
                "change_kind": "PUBLIC_POLICY", "event_polarity": "AFFIRMED",
                "change_relation": "NEW_OR_CHANGED_STATE",
                "material_relation_span": claim_text, "new_state": claim_text,
            }),
            "disruption": ("ESSENTIAL_SERVICE_DISRUPTION", {
                "service_kind": "TRANSPORT", "event_polarity": "AFFIRMED",
                "duration_relation": "DISRUPTION_DURATION", "duration_minutes": "90",
                "affected_group": claim_text,
            }),
        }[kind]
        raw["package"].update({
            "substantive_new_information": [claim_text], "governed_claims": [claim],
            "qualification_evidence": [{
                "test": test, "governed_claim_id": claim["claim_id"],
                "test_evidence": witnesses, "policy_version": "newsroom.evid-012.v7",
            }],
        })
        body = claim_text.encode()
        base = replace(base, passages=(claim_text,), observation_digests=(digest_bytes(body),))
    rights = PublicationRightsAssessment.create(
        decision="PERMITTED", permitted_use="PUBLICATION_EVIDENCE",
        policy_digest="sha256:" + "d" * 64, evidence_digest="sha256:" + "e" * 64,
    )
    source = SimpleNamespace(**{**vars(source), "rights": rights})
    acquired = SimpleNamespace(**{
        **vars(acquired), "body": body, "body_digest": digest_bytes(body),
        "currentness_basis": "AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT",
        "text_only": True, "exclusion_signals": (), "licence_attribution": "fixture",
        "rights_eligibility_digest": rights_eligibility_digest(
            rights, body_digest=digest_bytes(body),
            transport_digest=acquired.transport_evidence_digest,
            exclusion_signals=(), text_only=True,
        ),
    })
    candidate = SimpleNamespace(
        candidate_id=base.candidate_id, version_id="fixture-candidate-version",
        governing_manifest=SimpleNamespace(canonical_digest="sha256:" + "f" * 64),
        canonical_bytes=canonical_json_bytes({"candidate_id": base.candidate_id}),
    )
    return candidate, base, source, acquired, raw


@pytest.mark.parametrize("mutation", ["valid", "relation", "witness", "duration_valid", "duration_invalid"])
def test_assessor_qualification_witnesses_match_admission_helpers(
    retained_22589_assessment, mutation,
):
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment,
        kind=("disruption" if mutation.startswith("duration") else "policy" if mutation == "witness" else "deadline"),
    )
    evidence = raw["package"]["qualification_evidence"][0]["test_evidence"]
    if mutation == "relation":
        evidence["material_relation_span"] = "deadline"
    elif mutation == "witness":
        evidence["new_state"] = "invented changed state"
    elif mutation == "duration_invalid":
        evidence["duration_minutes"] = "120"
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {})
    if mutation in {"valid", "duration_valid"}:
        result = AutonomousNativeEvidenceAssessor._validated_execution(
            execution, candidate, base, (source,), (acquired,),
        )
        assert len(result.qualification_evidence) == 1
    else:
        with pytest.raises(EvidencePackageError, match="qualification"):
            AutonomousNativeEvidenceAssessor._validated_execution(
                execution, candidate, base, (source,), (acquired,),
            )


@pytest.mark.parametrize(
    "prior_contract",
    ["newsroom.native-evidence-assessor.v15", "newsroom.native-evidence-assessor.v16"],
)
@pytest.mark.parametrize("cached_only", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_retained_qualification_validation_controls_existing_fresh_attempt(
    tmp_path, monkeypatch, retained_22589_assessment, prior_contract, cached_only, valid,
):
    from newsroom.control_plane import native_assessor as module

    _use_historical_v16(monkeypatch)
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment, kind="deadline" if valid else "retained",
    )
    current_contract = module._V16_PRODUCER_VERSION
    valid_text = canonical_json_bytes(
        raw if valid else {"package": _model_package_value(base)}
    ).decode()
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {
        "usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 1,
        "cached_read_tokens": 0, "cached_write_tokens": 0,
        "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2,
    })
    monkeypatch.setattr(module, "VERSION", prior_contract)
    monkeypatch.setattr(
        module,
        "SYSTEM",
        module._V15_SYSTEM
        if prior_contract == module._V15_PRODUCER_VERSION
        else module._V16_SYSTEM,
    )
    monkeypatch.setattr(
        module,
        "CONTEXT_MANIFEST_SCHEMA_VERSION",
        "newsroom.native-evidence-assessor.context-manifest.v1"
        if prior_contract == module._V15_PRODUCER_VERSION
        else "newsroom.native-evidence-assessor.context-manifest.v2",
    )
    service, prior_usage = _usage(tmp_path, monkeypatch)
    # Seed the actual old acceptance shape without invoking an older validator:
    # a settled, exact retained raw result whose structural contract was accepted.
    allocation = prior_usage.begin(candidate, base, "retained assessor fixture")
    dispatched = prior_usage.mark_dispatch(allocation)
    assert prior_usage.retain_result(allocation, execution, dispatch_at=dispatched)
    prior_usage.complete(allocation, outcome="ASSESSOR_ACCEPTED", execution=execution,
                         provider_dispatched=True, dispatch_at=dispatched)
    with sqlite3.connect(service.path) as retained:
        original_allocation = retained.execute(
            "SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?",
            (allocation.invocation_id,),
        ).fetchone()
        original_result = retained.execute(
            "SELECT payload_digest,payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'",
        ).fetchone()
    assert json.loads(original_allocation[0])["prompt_contract_version"] == prior_contract
    monkeypatch.setattr(module, "VERSION", current_contract)
    monkeypatch.setattr(module, "SYSTEM", module._V16_SYSTEM)
    monkeypatch.setattr(
        module,
        "CONTEXT_MANIFEST_SCHEMA_VERSION",
        "newsroom.native-evidence-assessor.context-manifest.v2",
    )
    _, usage = _usage(tmp_path, monkeypatch)
    calls = []

    def dispatch(_prompt):
        calls.append("fixture-dispatch")
        return NativeAssessmentExecution(valid_text, execution.usage)

    assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=usage, dispatch_fence=nullcontext)
    old = prior_contract != current_contract
    for _ in range(2):
        if not valid and (cached_only or not old):
            with pytest.raises(NativeEvidenceHold, match="ASSESSOR_QUALIFICATION_CONTRACT_HOLD"):
                assessor.assess_with_boundary(candidate, base, (source,), (acquired,),
                                              before_dispatch=None, cached_only=cached_only)
        else:
            result = assessor.assess_with_boundary(candidate, base, (source,), (acquired,),
                                                  before_dispatch=None, cached_only=cached_only)
            if valid:
                assert dict(result.qualification_evidence[0].test_evidence)["reader_action"] == "The deadline changed."
            else:
                assert result.qualification_evidence == ()
    expected_dispatches = int(old and not valid and not cached_only)
    assert len(calls) == expected_dispatches
    with sqlite3.connect(service.path) as retained:
        assert retained.execute("SELECT COUNT(*) FROM model_invocation_allocations").fetchone() == (1 + expected_dispatches,)
        assert retained.execute("SELECT COUNT(*) FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchone() == (1 + expected_dispatches,)
        assert retained.execute(
            "SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?",
            (allocation.invocation_id,),
        ).fetchone() == original_allocation
        assert retained.execute(
            "SELECT payload_digest,payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT' "
            "AND json_extract(payload_json,'$.invocation_id')=?",
            (allocation.invocation_id,),
        ).fetchone() == original_result


@pytest.mark.parametrize("rendering_valid", [False, True])
@pytest.mark.parametrize(("source_prefix", "old_state"), [
    ("", False), ("Officials deny that ", False), ("Subject to approval, ", False),
    ("Officials propose that ", False), ("Officials deny that\n", False), ("", True),
    pytest.param("Subject to approval; ", False, id="conditional-semicolon"),
    pytest.param("Officials deny the following; ", False, id="denial-semicolon"),
    pytest.param("須經批准；", False, id="conditional-fullwidth-semicolon"),
])
def test_cached_operational_replacement_revalidates_without_dispatch_or_relabelling(
    tmp_path, monkeypatch, retained_22589_assessment, rendering_valid, source_prefix, old_state,
):
    _use_historical_v16(monkeypatch)
    from newsroom.control_plane.native_assessor import assessment_revalidation_due
    from newsroom.control_plane.native_composition import ASSESSMENT_CONTRACT_VERSION
    from newsroom.control_plane.native_evidence import rights_eligibility_digest
    from newsroom.tests.test_operational_replacement_qualification import REPLACEMENT

    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment, kind="policy",
    )
    claim = raw["package"]["governed_claims"][0]
    rendering = "改動以新學徒評核方式取代終期評核（EPA），讓評核在整個學徒期進行，而非只在期末進行。"
    if not rendering_valid:
        rendering += " apprenticeship assessment"
    claim.update(claim=REPLACEMENT, supporting_excerpt=REPLACEMENT,
                 rendered_assertion_zh_hant_hk=rendering)
    raw["package"]["substantive_new_information"] = [REPLACEMENT]
    raw["package"]["qualification_evidence"][0]["test_evidence"].update(
        material_relation_span=REPLACEMENT,
        new_state="end-point assessment (EPA)" if old_state else REPLACEMENT,
    )
    source_text = source_prefix + REPLACEMENT
    body = source_text.encode()
    base = replace(base, passages=(source_text,), observation_digests=(digest_bytes(body),))
    acquired = SimpleNamespace(**{
        **vars(acquired), "body": body, "body_digest": digest_bytes(body),
        "rights_eligibility_digest": rights_eligibility_digest(
            source.rights, body_digest=digest_bytes(body),
            transport_digest=acquired.transport_evidence_digest,
            exclusion_signals=(), text_only=True,
        ),
    })
    facts = {
        "reason": "ASSESSOR_QUALIFICATION_CONTRACT_HOLD",
        "assessment_contract_version": ASSESSMENT_CONTRACT_VERSION.rsplit("+", 1)[0],
    }
    assert assessment_revalidation_due(facts, ASSESSMENT_CONTRACT_VERSION)
    service, usage = _usage(tmp_path, monkeypatch)
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {
        "usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 1,
        "cached_read_tokens": 0, "cached_write_tokens": 0,
        "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2,
    })
    allocation = usage.begin(candidate, base, "retained operational replacement fixture")
    dispatch_at = usage.mark_dispatch(allocation)
    usage.retain_result(allocation, execution, dispatch_at=dispatch_at)
    usage.complete(
        allocation, outcome="ASSESSOR_VALIDATION_FAILED", execution=execution,
        provider_dispatched=True, dispatch_at=dispatch_at,
        failure_class="ASSESSMENT_VALIDATION_FAILED",
    )

    def retained_rows():
        with sqlite3.connect(service.path) as connection:
            return (
                connection.execute("SELECT record_json FROM model_invocation_allocations").fetchall(),
                connection.execute("SELECT record_json FROM model_invocation_terminals").fetchall(),
                connection.execute("SELECT payload_digest,payload_json FROM ledger").fetchall(),
            )

    before = retained_rows()
    assessor = AutonomousNativeEvidenceAssessor(
        lambda *_: pytest.fail("consumer-only revalidation dispatched"),
        usage=usage, dispatch_fence=nullcontext,
    )
    for _ in range(2):
        if rendering_valid and not source_prefix and not old_state:
            result = assessor.assess_with_boundary(
                candidate, base, (source,), (acquired,),
                before_dispatch=None, cached_only=True,
            )
            assert len(result.qualification_evidence) == 1
        else:
            reason = ("ASSESSOR_QUALIFICATION_CONTRACT_HOLD" if rendering_valid
                      else "ASSESSOR_RENDERING_CONTRACT_HOLD")
            with pytest.raises(NativeEvidenceHold, match=reason):
                assessor.assess_with_boundary(
                    candidate, base, (source,), (acquired,),
                    before_dispatch=None, cached_only=True,
                )
    assert retained_rows() == before
    assert json.loads(before[0][0][0])["prompt_contract_version"] == (
        native_assessor_module._V16_PRODUCER_VERSION
    )
    facts["assessment_contract_version"] = ASSESSMENT_CONTRACT_VERSION
    assert not assessment_revalidation_due(facts, ASSESSMENT_CONTRACT_VERSION)


def test_native_assessor_retains_precise_qualification_contract_hold(
    tmp_path, monkeypatch,
) -> None:
    _use_historical_v16(monkeypatch)
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    package = _model_package_value(base)
    package["qualification_evidence"] = [{
        "test": "OFFICIAL_ACTION_OR_DEADLINE",
        "governed_claim_id": "missing-claim",
        "test_evidence": {
            "action_class": "OFFICIAL_DEADLINE",
            "event_polarity": "AFFIRMED",
            "action_relation": "NEW_OR_CHANGED_OFFICIAL_ACTION",
            "material_relation_span": "deadline",
            "reader_action": "check deadline",
        },
        "policy_version": "newsroom.evid-012.v7",
    }]
    _service, usage = _usage(tmp_path, monkeypatch)

    with pytest.raises(
        NativeEvidenceHold, match="ASSESSOR_QUALIFICATION_CONTRACT_HOLD"
    ):
        AutonomousNativeEvidenceAssessor(
            lambda _prompt: NativeAssessmentExecution(
                canonical_json_bytes({"package": package}).decode(),
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
            ),
            usage=usage,
            dispatch_fence=nullcontext,
        )(candidate, base, (), ())
    connection.close()


def _usage(tmp_path, monkeypatch):
    current = (
        native_assessor_module.VERSION in {
            native_assessor_module._V20_PRODUCER_VERSION,
            native_assessor_module._V21_PRODUCER_VERSION,
            native_assessor_module._V22_PRODUCER_VERSION,
            native_assessor_module._REFERENCE_PRODUCER_VERSION,
        }
    )
    if not current:
        # Historical fixtures retain their model, reasoning and output ceiling.
        reasoning = (
            "medium" if native_assessor_module.VERSION == native_assessor_module._V19_PRODUCER_VERSION
            else CONT_PRIMARY_REASONING
        )
        monkeypatch.setattr(native_assessor_module, "MODEL", CONT_PRIMARY_MODEL)
        monkeypatch.setattr(native_assessor_module, "REASONING", reasoning)
        monkeypatch.setattr(
            native_assessor_module, "COMMAND_FLAGS", _grok_command_flags(reasoning),
        )
    monkeypatch.setattr(
        "newsroom.control_plane.native_assessor.read_grok_command_semantic_version",
        lambda **_kwargs: "1.0.8",
    )
    monkeypatch.setattr(
        "newsroom.control_plane.native_assessor.cont_writer_implementation_identity",
        lambda: (REVISION, True),
    )
    service = ModelUsageService(str(tmp_path / "usage.sqlite3"))
    connect(service.path).close()
    policy = InvocationEfficiencyPolicy.create(
        policy_id="native-assessor-policy",
        version="v1",
        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
        provider="grok-build-cli",
        route="NATIVE_EVIDENCE_ASSESSOR",
        model=native_assessor_module.MODEL,
        reasoning=native_assessor_module.REASONING,
        one_turn=True,
        exact_input=True,
        skills_enabled=False,
        tools_enabled=False,
        mcp_enabled=False,
        prior_message_count=0,
        command_semantic_version="1.0.8",
        command_flags=native_assessor_module.COMMAND_FLAGS,
        context_manifest_schema_version=(
            native_assessor_module.CONTEXT_MANIFEST_SCHEMA_VERSION
        ),
        disabled_capabilities=CONT_DISABLED_CAPABILITIES,
        implementation_revision=REVISION,
        max_prompt_bytes=1_000_000,
        max_context_tokens=100_000,
        max_output_tokens=None if current else 10_000,
        max_total_tokens=100_000,
        prompt_contract_version=native_assessor_module.VERSION,
        output_schema_digest=native_assessor_module.PROVIDER_SCHEMA_DIGEST,
        allowed_context_identities=(CONTEXT_IDENTITY,),
        allowed_config_identities=(CONFIG_IDENTITY,),
        hard_estimate_ceiling_tokens=100_000,
        evidence_digest="sha256:" + "a" * 64,
        qualified=True,
    )
    return service, NativeAssessmentUsage(
        service, policy, clock=lambda: datetime(2026, 9, 8, tzinfo=UTC)
    )


def test_native_assessor_input_bound_has_planning_headroom_not_an_output_limit(
    tmp_path, monkeypatch,
) -> None:
    _service, usage = _usage(tmp_path, monkeypatch)
    policy = usage._policy
    bound = native_assessment_input_bound(policy)
    assert bound['version'] == INPUT_BOUND_VERSION
    assert CONTEXT_MANIFEST_SCHEMA_VERSION.endswith('.v3')
    assert VERSION.endswith('.v23')
    assert bound['system_digest'] == digest_bytes(SYSTEM.encode('utf-8'))
    assert bound['system_bytes'] == len(SYSTEM.encode('utf-8'))
    assert bound['schema_digest'] == PROVIDER_SCHEMA_DIGEST
    assert bound['schema_bytes'] == len(canonical_json_bytes(PROVIDER_SCHEMA))
    assert bound['framing_reserve_tokens'] == 16_384
    assert policy.max_output_tokens is None
    assert bound['output_reserve_tokens'] == 10_000
    assert bound['output_limit_enforced'] is False
    assert bound['output_reserve_basis'] == 'PLANNING_ONLY'
    fixed = bound['system_bytes'] + bound['schema_bytes'] + 16_384
    assert bound['max_request_bytes'] == min(
        policy.max_prompt_bytes, policy.max_context_tokens - fixed,
        policy.max_total_tokens - fixed - bound['output_reserve_tokens'],
    )
    assert bound['max_request_bytes'] == 61_726 - (
        len(SYSTEM.encode()) - len(native_assessor_module._V21_SYSTEM.encode())
    )
    assert native_assessment_input_bound(replace(
        policy, max_prompt_bytes=56_464,
    ))['max_request_bytes'] == 56_464
    assert bound['bound_digest'] == digest_canonical({
        key: value for key, value in bound.items() if key != 'bound_digest'
    })
    assert native_assessment_input_bound(policy) == bound
    assert native_assessment_input_bound(replace(
        policy, max_context_tokens=policy.max_context_tokens - 1,
    ))['max_request_bytes'] <= bound['max_request_bytes']
    assert native_assessment_input_bound(replace(
        policy, max_total_tokens=policy.max_total_tokens - 1,
    ))['max_request_bytes'] < bound['max_request_bytes']


@pytest.mark.parametrize('request_bytes', [26_051, 19_254, 397_776])
def test_native_assessor_exact_request_bound_precedes_allocation(
    tmp_path, monkeypatch, request_bytes,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    bound = native_assessment_input_bound(usage._policy)
    assert 26_051 <= bound['max_request_bytes'] < 397_776
    try:
        if request_bytes > bound['max_request_bytes']:
            with pytest.raises(NativeEvidenceHold) as held:
                usage.begin(candidate, base, 'x' * request_bytes)
            assert held.value.reason_code == 'ASSESSOR_EXACT_INPUT_BOUND_HOLD'
            with sqlite3.connect(service.path) as retained:
                for table in (
                    'model_work_envelopes', 'model_invocation_context_manifests',
                    'model_invocation_allocations', 'model_transport_observations',
                ):
                    assert retained.execute(f'SELECT count(*) FROM {table}').fetchone() == (0,)
        else:
            allocation = usage.begin(candidate, base, 'x' * request_bytes)
            assert allocation.prompt_bytes == request_bytes
            with sqlite3.connect(service.path) as retained:
                raw, = retained.execute(
                    'SELECT record_json FROM model_invocation_context_manifests'
                ).fetchone()
                manifest = json.loads(raw)
                assert manifest['input_bound'] == bound
                assert manifest['schema_version'] == CONTEXT_MANIFEST_SCHEMA_VERSION
                assert retained.execute('SELECT count(*) FROM model_transport_observations').fetchone() == (0,)
    finally:
        connection.close()


def test_native_assessor_exact_byte_ceiling_and_plus_one(tmp_path, monkeypatch) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    maximum = native_assessment_input_bound(usage._policy)['max_request_bytes']
    try:
        with pytest.raises(NativeEvidenceHold) as held:
            usage.begin(candidate, base, 'é' * ((maximum // 2) + 1))
        assert held.value.reason_code == 'ASSESSOR_EXACT_INPUT_BOUND_HOLD'
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT count(*) FROM model_work_envelopes').fetchone() == (0,)
        exact = 'é' * (maximum // 2) + ('x' if maximum % 2 else '')
        allocation = usage.begin(candidate, base, exact)
        assert allocation.prompt_bytes == maximum
    finally:
        connection.close()


def test_assessor_rejects_source_lower_bound_before_building_reference_view(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    maximum = native_assessment_input_bound(usage._policy)["max_request_bytes"]
    oversized = replace(
        base,
        passages=("x" * (maximum + 1),),
        observation_digests=(digest_bytes(b"x" * (maximum + 1)),),
    )
    monkeypatch.setattr(
        native_assessor_module,
        "build_source_view",
        lambda *_args: pytest.fail("over-bound source built a reference view"),
    )
    assessor = AutonomousNativeEvidenceAssessor(
        lambda _prompt: pytest.fail("over-bound source dispatched provider"),
        usage=usage,
        dispatch_fence=nullcontext,
    )
    try:
        with pytest.raises(NativeEvidenceHold) as held:
            assessor.assess_with_boundary(
                candidate, oversized, (), (), before_dispatch=None
            )
        assert held.value.reason_code == "ASSESSOR_EXACT_INPUT_BOUND_HOLD"
        with sqlite3.connect(service.path) as retained:
            assert retained.execute(
                "SELECT COUNT(*) FROM model_work_envelopes"
            ).fetchone() == (0,)
            assert retained.execute(
                "SELECT COUNT(*) FROM model_invocation_allocations"
            ).fetchone() == (0,)
    finally:
        connection.close()


def test_native_assessor_uses_exact_candidate_and_base_without_ambient_context(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    calls = []
    fence_active = False
    usage_service, usage = _usage(tmp_path, monkeypatch)

    def dispatch(prompt):
        assert fence_active
        with sqlite3.connect(usage_service.path) as retained:
            assert retained.execute(
                "SELECT state FROM model_transport_observations"
            ).fetchall() == [("DISPATCH_STARTED",)]
        calls.append(prompt)
        return NativeAssessmentExecution(
            canonical_json_bytes(_empty_reference_result()).decode(),
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

    @contextmanager
    def fence():
        nonlocal fence_active
        fence_active = True
        try:
            yield
        finally:
            fence_active = False

    boundaries = []

    def before_dispatch():
        with sqlite3.connect(usage_service.path) as retained:
            assert retained.execute(
                "SELECT COUNT(*) FROM model_invocation_allocations"
            ).fetchone() == (1,)
            assert retained.execute(
                "SELECT COUNT(*) FROM model_transport_observations"
            ).fetchone() == (0,)
        boundaries.append("ASSESSMENT_STARTED")

    result = AutonomousNativeEvidenceAssessor(
        dispatch, usage=usage, dispatch_fence=fence
    ).assess_with_boundary(
        candidate, base, (), (), before_dispatch=before_dispatch
    )
    assert fence_active is False
    assert boundaries == ["ASSESSMENT_STARTED"]
    request = json.loads(calls[0])
    assert (
        request["candidate_version"]["version"]["version_id"]
        == candidate.version_id
    )
    expected_base = evidence_package_value(base)
    expected_base.pop("passages")
    assert request["base_package"] == expected_base
    assert request["sources"] == []
    assert request["output_schema_digest"] == PROVIDER_SCHEMA_DIGEST
    assert result.governed_claims == ()
    with sqlite3.connect(usage_service.path) as retained:
        assert retained.execute(
            "SELECT outcome FROM model_invocation_terminals"
        ).fetchall() == [("ASSESSOR_ACCEPTED",)]

    bad = AutonomousNativeEvidenceAssessor(
        lambda _: NativeAssessmentExecution('{"package": {}}', {})
    )
    with pytest.raises(EvidencePackageError):
        bad(candidate, base, (), ())
    connection.close()


def test_native_assessor_command_version_is_observed_without_becoming_a_gate(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "newsroom.control_plane.native_assessor.read_grok_command_semantic_version",
        lambda: "1.0.9",
    )
    allocation = usage.begin(candidate, base, "retained assessor prompt")
    with sqlite3.connect(service.path) as retained:
        assert retained.execute(
            "SELECT count(*) FROM model_work_envelopes"
        ).fetchone() == (1,)
        assert retained.execute(
            "SELECT count(*) FROM model_invocation_allocations"
        ).fetchone() == (1,)
        assert retained.execute(
            "SELECT count(*) FROM model_transport_observations"
        ).fetchone() == (0,)
        manifest = json.loads(retained.execute(
            "SELECT record_json FROM model_invocation_context_manifests "
            "WHERE context_manifest_digest=?",
            (allocation.context_manifest_digest,),
        ).fetchone()[0])
        assert manifest["command_semantic_version"] == "1.0.9"
    connection.close()


def test_native_assessor_pre_dispatch_recovery_requires_zero_exact_envelopes(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    _service, usage = _usage(tmp_path, monkeypatch)

    proof = usage.retained_pre_dispatch_failure(candidate)
    assert proof is not None
    assert proof.candidate_id == candidate.candidate_id
    assert proof.candidate_version_id == candidate.version_id
    allocation = usage.begin(candidate, base, "retained assessor prompt")
    assert usage.retained_pre_dispatch_failure(candidate) is None
    other = SimpleNamespace(
        candidate_id="other-candidate",
        version_id="other-version",
        governing_manifest=SimpleNamespace(
            canonical_digest=candidate.governing_manifest.canonical_digest
        ),
    )
    assert usage.retained_pre_dispatch_failure(other) is not None
    with sqlite3.connect(_service.path) as usage_connection:
        usage_connection.execute(
            "UPDATE model_invocation_allocations SET envelope_id='orphan'"
        )
    assert usage.retained_pre_dispatch_failure(other) is None
    with sqlite3.connect(_service.path) as usage_connection:
        usage_connection.execute(
            "UPDATE model_invocation_allocations SET envelope_id=?",
            (allocation.envelope_id,),
        )
    assert usage.retained_pre_dispatch_failure(other) is not None
    dispatch_at = usage.mark_dispatch(allocation)
    with sqlite3.connect(_service.path) as usage_connection:
        usage_connection.execute(
            "UPDATE model_transport_observations SET invocation_id='orphan'"
        )
    assert usage.retained_pre_dispatch_failure(other) is None
    with sqlite3.connect(_service.path) as usage_connection:
        usage_connection.execute(
            "UPDATE model_transport_observations SET invocation_id=?",
            (allocation.invocation_id,),
        )
    usage.complete(
        allocation,
        outcome="ASSESSOR_PROVIDER_FAILED",
        execution=NativeAssessmentExecution(
            "provider response",
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
        ),
        provider_dispatched=True,
        dispatch_at=dispatch_at,
        failure_class="SYSTEMIC",
    )
    assert usage.retained_pre_dispatch_failure(other) is not None
    with sqlite3.connect(_service.path) as usage_connection:
        usage_connection.execute("PRAGMA foreign_keys=OFF")
        usage_connection.execute("DELETE FROM model_transport_observations")
        usage_connection.execute("DELETE FROM model_invocation_allocations")
        usage_connection.execute("DELETE FROM model_work_envelopes")
        assert usage_connection.execute(
            "SELECT COUNT(*) FROM model_invocation_terminals"
        ).fetchone() == (1,)
    assert usage.retained_pre_dispatch_failure(other) is None
    connection.close()


def test_native_assessor_resumes_only_exact_empty_envelope_when_route_eligible(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    from newsroom.control_plane.native_assessor import _assessment_cycle_id
    envelope = WorkEnvelope.create(
        cycle_id=_assessment_cycle_id(candidate.version_id, base.digest, VERSION),
        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
        admitted_at=datetime(2026, 9, 7, tzinfo=UTC),
        admission_decision_id=None, candidate_id=candidate.candidate_id,
        hypothesis_digest=candidate.governing_manifest.canonical_digest,
        evidence_package_digest=base.digest, ingest_id=None, graphiti_attempt_id=None,
    )
    service.open_envelope(envelope)
    try:
        assert usage.retained_assessments(candidate, base) == ()
        assert usage.retained_pre_dispatch_failure(candidate) is not None
        allocation = usage.begin(candidate, base, 'exact resumed request')
        assert allocation.envelope_id == envelope.envelope_id
        assert usage.retained_pre_dispatch_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT count(*) FROM model_work_envelopes').fetchone() == (1,)
            assert retained.execute('SELECT count(*) FROM model_invocation_allocations').fetchone() == (1,)
            assert retained.execute('SELECT count(*) FROM model_transport_observations').fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize('defect', ['wrong-manifest', 'wrong-cycle', 'route-open', 'different-base'])
def test_native_assessor_empty_envelope_mismatch_remains_unresolved(
    tmp_path, monkeypatch, defect,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    from newsroom.control_plane.native_assessor import _assessment_cycle_id
    retained_base_digest = digest_bytes(b'different base') if defect == 'different-base' else base.digest
    envelope = WorkEnvelope.create(
        cycle_id=(digest_bytes(b'wrong cycle') if defect == 'wrong-cycle' else
                  _assessment_cycle_id(candidate.version_id, retained_base_digest, VERSION)),
        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
        admitted_at=datetime(2026, 9, 7, tzinfo=UTC),
        admission_decision_id=None, candidate_id=candidate.candidate_id,
        hypothesis_digest=(digest_bytes(b'wrong manifest') if defect == 'wrong-manifest' else
                           candidate.governing_manifest.canonical_digest),
        evidence_package_digest=retained_base_digest,
        ingest_id=None, graphiti_attempt_id=None,
    )
    service.open_envelope(envelope)
    if defect == 'route-open':
        with service._connection() as retained:
            service._append_route_state(
                retained, route='NATIVE_EVIDENCE_ASSESSOR', state='OPEN',
                reason='QUOTA', invocation_id=None,
                recorded_at=datetime(2026, 9, 7, tzinfo=UTC),
            )
    try:
        if defect in {'wrong-manifest', 'wrong-cycle', 'route-open'}:
            assert usage.retained_pre_dispatch_failure(candidate) is None
        else:
            assert usage.retained_pre_dispatch_failure(candidate) is not None
        if defect != 'route-open':
            assert usage.retained_assessments(candidate, base) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT count(*) FROM model_invocation_allocations').fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("output", "outcome"),
    (
        (None, "ASSESSOR_PROVIDER_FAILED"),
        ('{"package":{}}', "ASSESSOR_VALIDATION_FAILED"),
        ('{', "ASSESSOR_VALIDATION_FAILED"),
        ('{"package":{},"package":{}}', "ASSESSOR_VALIDATION_FAILED"),
    ),
)
def test_native_assessor_retains_post_dispatch_failures(
    tmp_path,
    monkeypatch,
    output,
    outcome,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    dispatches = 0

    def dispatch(_request):
        nonlocal dispatches
        dispatches += 1
        if output is None:
            raise RuntimeError("provider broke")
        return NativeAssessmentExecution(
            output,
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

    with pytest.raises((RuntimeError, NativeEvidenceError)) as caught:
        AutonomousNativeEvidenceAssessor(
            dispatch, usage=usage, dispatch_fence=nullcontext
        )(
            candidate, base, (), ()
        )
    if outcome == "ASSESSOR_VALIDATION_FAILED":
        assert isinstance(caught.value, NativeEvidenceHold)
        assert caught.value.reason_code == "ASSESSOR_SOURCE_REFERENCE_HOLD"
    assert dispatches == 1

    with sqlite3.connect(service.path) as retained:
        terminal = json.loads(retained.execute(
            "SELECT record_json FROM model_invocation_terminals"
        ).fetchone()[0])
        assert terminal["outcome"] == outcome
        assert terminal["pre_dispatch_zero_proved"] is False
        assert terminal["dispatch_at"] is not None
        assert retained.execute(
            "SELECT state FROM model_transport_observations ORDER BY state"
        ).fetchall() == [("DISPATCH_STARTED",)]
        result_rows = retained.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'"
        ).fetchall()
        assert len(result_rows) == (0 if output is None else 1)
        assert retained.execute(
            "SELECT COUNT(*) FROM ledger "
            "WHERE kind='NATIVE_ASSESSMENT_MATERIALISATION'"
        ).fetchone() == (0,)
        if output is not None:
            diagnostic = json.loads(result_rows[0][0])
            assert diagnostic["result_text"] == output
            assert diagnostic["result_bytes"] == len(output.encode())
    proof = usage.retained_output_contract_failure(candidate)
    if outcome == "ASSESSOR_VALIDATION_FAILED":
        assert proof is not None
        assert proof.invocation_id == terminal["invocation_id"]
        assert proof.terminal_digest == terminal["terminal_digest"]
        monkeypatch.setattr(
            "newsroom.control_plane.native_assessor.SCHEMA_DIGEST",
            "sha256:" + "9" * 64,
        )
        assert usage.retained_output_contract_failure(candidate) is not None
        wrong_candidate = SimpleNamespace(
            candidate_id="wrong-candidate",
            version_id=candidate.version_id,
            governing_manifest=candidate.governing_manifest,
        )
        assert usage.retained_output_contract_failure(wrong_candidate) is None
        with sqlite3.connect(service.path) as retained:
            retained.execute(
                "UPDATE model_invocation_policies SET qualified=0"
            )
        assert usage.retained_output_contract_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            retained.execute(
                "UPDATE model_invocation_policies SET qualified=1"
            )
        assert usage.retained_output_contract_failure(candidate) is not None
        with sqlite3.connect(service.path) as retained:
            original_policy = retained.execute(
                "SELECT record_json FROM model_invocation_policies"
            ).fetchone()[0]
            changed_policy = json.loads(original_policy)
            changed_policy["max_total_tokens"] += 1
            retained.execute(
                "UPDATE model_invocation_policies SET record_json=?",
                (json.dumps(changed_policy),),
            )
        assert usage.retained_output_contract_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            retained.execute(
                "UPDATE model_invocation_policies SET record_json=?",
                (original_policy,),
            )
        assert usage.retained_output_contract_failure(candidate) is not None
        with sqlite3.connect(service.path) as retained:
            coerced_policy = json.loads(original_policy)
            coerced_policy["qualified"] = 1
            retained.execute(
                "UPDATE model_invocation_policies SET record_json=?",
                (json.dumps(coerced_policy),),
            )
        assert usage.retained_output_contract_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            retained.execute(
                "UPDATE model_invocation_policies SET record_json=?",
                (original_policy,),
            )
        assert usage.retained_output_contract_failure(candidate) is not None
        with sqlite3.connect(service.path) as retained:
            retained.execute(
                "UPDATE model_provider_telemetry "
                "SET provider_telemetry_digest=?",
                ("sha256:" + "f" * 64,),
            )
        assert usage.retained_output_contract_failure(candidate) is None
    else:
        assert proof is None
    connection.close()


def test_native_assessor_result_diagnostic_is_bounded_and_replay_safe(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    allocation = usage.begin(candidate, base, "exact request")
    dispatch_at = usage.mark_dispatch(allocation)
    execution = NativeAssessmentExecution('{"malformed":true}', {})

    assert usage.retain_result(
        allocation, execution, dispatch_at=dispatch_at
    ) is True
    assert usage.retain_result(
        allocation, execution, dispatch_at=dispatch_at
    ) is True
    with pytest.raises(
        NativeEvidenceError, match="conflicting native assessment result replay"
    ):
        usage.retain_result(
            allocation,
            NativeAssessmentExecution('{"different":true}', {}),
            dispatch_at=dispatch_at,
        )
    with sqlite3.connect(service.path) as retained:
        result = json.loads(retained.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'"
        ).fetchone()[0])
        assert result["result_text"] == execution.text
        assert result["result_digest"] == digest_bytes(execution.text.encode())
        assert result["retention_outcome"] == "RETAINED"
        assert retained.execute(
            "SELECT COUNT(*) FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'"
        ).fetchone()[0] == 1
        assert retained.execute(
            "SELECT COUNT(*) FROM model_invocation_terminals"
        ).fetchone()[0] == 0
    connection.close()


def test_native_assessor_result_diagnostic_rejects_oversized_output(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    allocation = usage.begin(candidate, base, "exact request")
    dispatch_at = usage.mark_dispatch(allocation)
    text = "x" * (_MAX_RETAINED_RESULT_BYTES + 1)

    assert usage.retain_result(
        allocation, NativeAssessmentExecution(text, {}), dispatch_at=dispatch_at
    ) is False
    with sqlite3.connect(service.path) as retained:
        result = json.loads(retained.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'"
        ).fetchone()[0])
    assert result["result_text"] is None
    assert result["result_bytes"] == len(text)
    assert result["result_digest"] == digest_bytes(text.encode())
    assert result["retention_outcome"] == "OVERSIZED"
    connection.close()


def test_native_assessor_oversized_result_becomes_accounted_contract_hold(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    output = "x" * (_MAX_RETAINED_RESULT_BYTES + 1)
    provider_usage = {
        "usage_basis": "PROVIDER_REPORTED",
        "input_tokens": 1,
        "output_tokens": 1,
        "cached_read_tokens": 0,
        "cached_write_tokens": 0,
        "reasoning_tokens": 0,
        "context_tokens": 1,
        "total_tokens": 2,
    }

    with pytest.raises(
        NativeEvidenceHold, match="ASSESSOR_OUTPUT_CONTRACT_HOLD"
    ):
        AutonomousNativeEvidenceAssessor(
            lambda _request: NativeAssessmentExecution(output, provider_usage),
            usage=usage,
            dispatch_fence=nullcontext,
        )(candidate, base, (), ())

    with sqlite3.connect(service.path) as retained:
        assert retained.execute(
            "SELECT outcome FROM model_invocation_terminals"
        ).fetchall() == [("ASSESSOR_VALIDATION_FAILED",)]
        result = json.loads(retained.execute(
            "SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'"
        ).fetchone()[0])
        assert result["retention_outcome"] == "OVERSIZED"
        assert result["result_text"] is None
    connection.close()


def test_native_work_envelopes_reject_unrelated_authority_ids() -> None:
    common = {
        "cycle_id": "cycle-1",
        "admitted_at": datetime(2026, 9, 8, tzinfo=UTC),
        "admission_decision_id": None,
        "candidate_id": None,
        "hypothesis_digest": None,
        "evidence_package_digest": None,
        "ingest_id": "passage-1",
        "graphiti_attempt_id": "not-a-native-graphiti-attempt",
    }
    with pytest.raises(ModelUsageIntegrityError):
        WorkEnvelope.create(
            workload_class=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
            **common,
        )
    with pytest.raises(ModelUsageIntegrityError):
        WorkEnvelope.create(
            workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
            **{
                **common,
                "admission_decision_id": "not-an-assessor-admission",
                "candidate_id": "candidate-1",
                "hypothesis_digest": "sha256:" + "a" * 64,
                "evidence_package_digest": "sha256:" + "b" * 64,
                "ingest_id": None,
                "graphiti_attempt_id": None,
            },
        )


def test_inflight_native_assessor_is_not_a_retained_contract_failure(
    tmp_path, monkeypatch,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    _service, usage = _usage(tmp_path, monkeypatch)

    usage.begin(candidate, base, "in-flight assessor request")

    assert usage.retained_output_contract_failure(candidate) is None
    connection.close()


@pytest.mark.parametrize("new_contract", (False, True))
def test_retained_assessment_revalidation_reuses_output_without_provider(tmp_path, monkeypatch, new_contract):
    import newsroom.control_plane.native_assessor as module

    _use_historical_v16(monkeypatch)
    if new_contract:
        monkeypatch.setattr(module, "VERSION", module._V15_PRODUCER_VERSION)
        monkeypatch.setattr(module, "SYSTEM", module._V15_SYSTEM)
        monkeypatch.setattr(
            module, "CONTEXT_MANIFEST_SCHEMA_VERSION",
            "newsroom.native-evidence-assessor.context-manifest.v1",
        )
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    service, usage = _usage(tmp_path, monkeypatch)
    calls = []

    def dispatch(_prompt):
        calls.append("provider")
        return NativeAssessmentExecution(
            canonical_json_bytes({"package": _model_package_value(base)}).decode(),
            {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1,
             "output_tokens": 1, "cached_read_tokens": 0, "cached_write_tokens": 0,
             "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2},
        )

    assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=usage, dispatch_fence=nullcontext)
    first = assessor(candidate, base, (), ())
    if not new_contract:
        _, reopened_usage = _usage(tmp_path, monkeypatch)
        reopened_assessor = AutonomousNativeEvidenceAssessor(
            dispatch, usage=reopened_usage, dispatch_fence=nullcontext,
        )
        assert reopened_assessor.assess_with_boundary(
            candidate, base, (), (), before_dispatch=None, cached_only=True,
        ) == first
    if new_contract:
        monkeypatch.setattr(module, "VERSION", module._V16_PRODUCER_VERSION)
        monkeypatch.setattr(module, "SYSTEM", module._V16_SYSTEM)
        monkeypatch.setattr(
            module, "CONTEXT_MANIFEST_SCHEMA_VERSION",
            "newsroom.native-evidence-assessor.context-manifest.v2",
        )
        _, usage = _usage(tmp_path, monkeypatch)
        assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=usage, dispatch_fence=nullcontext)
    assert assessor(candidate, base, (), ()) == first
    assert assessor.assess_with_boundary(
        candidate, base, (), (), before_dispatch=None, cached_only=True,
    ) == first
    assert calls == ["provider"]
    with sqlite3.connect(service.path) as retained:
        assert retained.execute("SELECT COUNT(*) FROM model_invocation_allocations").fetchone() == (1,)
        original_digest = retained.execute("SELECT payload_digest FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchone()[0]
        retained.execute("UPDATE ledger SET payload_digest='corrupt' WHERE kind='NATIVE_ASSESSMENT_RESULT'")
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_UNRESOLVED_HOLD"):
        assessor(candidate, base, (), ())
    assert calls == ["provider"]
    with sqlite3.connect(service.path) as retained:
        retained.execute("UPDATE ledger SET payload_digest=? WHERE kind='NATIVE_ASSESSMENT_RESULT'", (original_digest,))
        retained.execute("UPDATE model_work_envelopes SET record_json=json_set(record_json,'$.candidate_id','hidden')")
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_UNRESOLVED_HOLD"):
        assessor(candidate, base, (), ())
    assert calls == ["provider"]
    connection.close()


def test_consumer_only_revalidation_requires_exact_cached_input(
    tmp_path, monkeypatch,
) -> None:
    import newsroom.control_plane.native_assessor as module

    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    calls = []

    def dispatch(_prompt):
        calls.append("provider")
        return NativeAssessmentExecution(
            canonical_json_bytes({"package": _model_package_value(base)}).decode(),
            {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1,
             "output_tokens": 1, "cached_read_tokens": 0, "cached_write_tokens": 0,
             "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(module, "VERSION", "newsroom.native-evidence-assessor.v11")
    monkeypatch.setattr(__import__(__name__, fromlist=["VERSION"]), "VERSION", module.VERSION)
    service, old_usage = _usage(tmp_path, monkeypatch)
    first = AutonomousNativeEvidenceAssessor(
        dispatch, usage=old_usage, dispatch_fence=nullcontext,
    )(candidate, base, (), ())
    monkeypatch.setattr(module, "VERSION", module._V15_PRODUCER_VERSION)
    monkeypatch.setattr(module, "SYSTEM", module._V15_SYSTEM)
    monkeypatch.setattr(
        module, "CONTEXT_MANIFEST_SCHEMA_VERSION",
        "newsroom.native-evidence-assessor.context-manifest.v1",
    )
    _, usage = _usage(tmp_path, monkeypatch)
    assessor = AutonomousNativeEvidenceAssessor(
        dispatch, usage=usage, dispatch_fence=nullcontext,
    )

    assert assessor.assess_with_boundary(
        candidate, base, (), (), before_dispatch=None, cached_only=True,
    ) == first
    changed = replace(base, passages=(base.passages[0] + " changed",))
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_INPUT_CHANGED_HOLD"):
        assessor.assess_with_boundary(
            candidate, changed, (), (), before_dispatch=None, cached_only=True,
        )
    assert calls == ["provider"]
    with sqlite3.connect(service.path) as retained:
        assert retained.execute(
            "SELECT COUNT(*) FROM model_invocation_allocations"
        ).fetchone() == (1,)
    connection.close()


def test_consumer_only_revalidation_without_retention_never_dispatches(
    tmp_path,
) -> None:
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    calls = []
    assessor = AutonomousNativeEvidenceAssessor(
        lambda _prompt: calls.append("provider"),
    )

    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_CACHE_MISSING_HOLD"):
        assessor.assess_with_boundary(
            candidate, base, (), (), before_dispatch=None, cached_only=True,
        )
    assert calls == []
    fallback = EvidenceAssessor(lambda *_args: calls.append("fallback"))
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_REVALIDATION_CACHE_MISSING_HOLD"):
        fallback.assess(candidate, base, (), (), cached_only=True)
    assert calls == []
    connection.close()


@pytest.mark.parametrize("prior_contract", (
    "newsroom.native-evidence-assessor.v6",
    "newsroom.native-evidence-assessor.v7",
    "newsroom.native-evidence-assessor.v8",
    "newsroom.native-evidence-assessor.v9",
    "newsroom.native-evidence-assessor.v10",
    "newsroom.native-evidence-assessor.v11",
    "newsroom.native-evidence-assessor.v12",
))
@pytest.mark.parametrize("settled", (True, False))
def test_superseded_assessor_allows_one_new_contract_attempt_only_after_settlement(
    tmp_path, monkeypatch, settled, prior_contract,
):
    import newsroom.control_plane.native_assessor as module

    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    monkeypatch.setattr(module, "VERSION", prior_contract)
    monkeypatch.setattr(__import__(__name__, fromlist=["VERSION"]), "VERSION", module.VERSION)
    service, old_usage = _usage(tmp_path, monkeypatch)
    execution = NativeAssessmentExecution(
        "not JSON",
        {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1,
         "output_tokens": 1, "cached_read_tokens": 0, "cached_write_tokens": 0,
         "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2},
    )
    if settled:
        with pytest.raises(NativeEvidenceHold):
            AutonomousNativeEvidenceAssessor(
                lambda _: execution, usage=old_usage, dispatch_fence=nullcontext,
            )(candidate, base, (), ())
    else:
        old_usage.begin(candidate, base, "unknown prior attempt")
    monkeypatch.setattr(module, "VERSION", "newsroom.native-evidence-assessor.v13")
    monkeypatch.setattr(__import__(__name__, fromlist=["VERSION"]), "VERSION", module.VERSION)
    _, new_usage = _usage(tmp_path, monkeypatch)
    calls = []

    def dispatch(_prompt):
        assert json.loads(_prompt)["prior_validation_feedback"] == {
            "reason": "ASSESSOR_OUTPUT_CONTRACT_HOLD",
            "prior_result_digest": digest_bytes(execution.text.encode()),
            "claims": [],
        }
        calls.append("provider")
        return execution

    assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=new_usage, dispatch_fence=nullcontext)
    for _ in range(2):
        with pytest.raises(NativeEvidenceHold):
            assessor(candidate, base, (), ())
    assert calls == (["provider"] if settled else [])
    with sqlite3.connect(service.path) as retained:
        assert retained.execute("SELECT COUNT(*) FROM model_invocation_allocations").fetchone() == (2 if settled else 1,)
    connection.close()


def test_assessor_feedback_retains_only_failed_output_diagnostics():
    from newsroom.control_plane.native_assessor import _validation_feedback

    claim = {
        "claim_id": "gc-withdrawn-fms",
        "claim": "We have also withdrawn the FMS comparison matrix page.",
        "rendered_assertion_zh_hant_hk": "我們亦已撤回 FMS comparison matrix 頁面。",
        "source_ids": ["UK-05"],
    }
    execution = NativeAssessmentExecution(
        json.dumps({"package": {"governed_claims": [claim]}}), {},
    )
    assert _validation_feedback(execution, "ASSESSOR_RENDERING_CONTRACT_HOLD") == {
        "reason": "ASSESSOR_RENDERING_CONTRACT_HOLD",
        "prior_result_digest": digest_bytes(execution.text.encode()),
        "claims": [{key: claim[key] for key in (
            "claim_id", "claim", "rendered_assertion_zh_hant_hk",
        )}],
    }


@pytest.mark.parametrize("text, organisations", (
    ("The Department for Education announced funding.", {"Department for Education"}),
    ("The Department announced funding.", set()),
    ("Authority Announces New Bank", set()),
    ("The Ministry of Justice and the Food Standards Agency", {"Ministry of Justice", "Food Standards Agency"}),
))
def test_rejected_organisation_prefix_does_not_hide_a_complete_name(text, organisations):
    assert {name for name, kind in bounded_named_entities(text) if kind == "ORGANISATION"} == organisations


def test_settled_rendering_failure_supplies_feedback_once_then_reuses_valid_result(
    tmp_path, monkeypatch, retained_22589_assessment,
):
    from newsroom.control_plane import native_assessor as module

    _use_historical_v16(monkeypatch)
    candidate, base, source, acquired, valid = _qualification_assessor_inputs(
        retained_22589_assessment, kind="deadline",
    )
    invalid = json.loads(canonical_json_bytes(valid))
    invalid["package"]["governed_claims"][0]["rendered_assertion_zh_hant_hk"] += " deadline changed"
    execution = NativeAssessmentExecution(canonical_json_bytes(invalid).decode(), {
        "usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 1,
        "cached_read_tokens": 0, "cached_write_tokens": 0,
        "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2,
    })
    current_contract = module._V16_PRODUCER_VERSION
    monkeypatch.setattr(module, "VERSION", module._V15_PRODUCER_VERSION)
    monkeypatch.setattr(module, "SYSTEM", module._V15_SYSTEM)
    monkeypatch.setattr(
        module, "CONTEXT_MANIFEST_SCHEMA_VERSION",
        "newsroom.native-evidence-assessor.context-manifest.v1",
    )
    service, prior_usage = _usage(tmp_path, monkeypatch)
    with pytest.raises(NativeEvidenceHold, match="ASSESSOR_RENDERING_CONTRACT_HOLD"):
        AutonomousNativeEvidenceAssessor(
            lambda _: execution, usage=prior_usage, dispatch_fence=nullcontext,
        )(candidate, base, (source,), (acquired,))

    monkeypatch.setattr(module, "VERSION", current_contract)
    monkeypatch.setattr(module, "SYSTEM", module._V16_SYSTEM)
    monkeypatch.setattr(
        module, "CONTEXT_MANIFEST_SCHEMA_VERSION",
        "newsroom.native-evidence-assessor.context-manifest.v2",
    )
    _, usage = _usage(tmp_path, monkeypatch)
    prompts = []

    def dispatch(prompt):
        prompts.append(json.loads(prompt))
        return NativeAssessmentExecution(canonical_json_bytes(valid).decode(), execution.usage)

    assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=usage, dispatch_fence=nullcontext)
    for _ in range(2):
        result = assessor(candidate, base, (source,), (acquired,))
        assert result.qualification_evidence
    assert len(prompts) == 1
    feedback = prompts[0]["prior_validation_feedback"]
    assert feedback["reason"] == "ASSESSOR_RENDERING_CONTRACT_HOLD"
    assert feedback["claims"][0]["rendered_assertion_zh_hant_hk"].endswith(" deadline changed")
    with sqlite3.connect(service.path) as retained:
        assert retained.execute("SELECT COUNT(*) FROM model_invocation_allocations").fetchone() == (2,)
        assert retained.execute("SELECT COUNT(*) FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchone() == (2,)


def test_named_entity_record_identity_binds_immutable_policy(monkeypatch):
    from newsroom.control_plane import native_assessor as module

    values = ("claim", "ETA", "OFFICIAL_TERM", "ETA")
    old = module._named_entity_record_id(*values)
    monkeypatch.setattr(module, "NAMED_ENTITY_POLICY_VERSION", "next-policy")
    new = module._named_entity_record_id(*values)
    assert new != old
    assert module._named_entity_record_id(*values) == new


@pytest.mark.parametrize("fault", (None, "old-state", "no-headline-proof", "source-drift"))
def test_source_bound_replacement_preserves_claims_but_only_proves_supported_qualifications(
    retained_22589_assessment, fault,
):
    # Actual retained result 29221: valid headline replacement; an additional
    # STATUS classifier is not evidence for the separate responsibility claim.
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment, kind="policy",
    )
    headline = "The reforms replaced end-point assessment with a new assessment model, now called apprenticeship assessment."
    background = "Responsibility for the overall apprenticeship programme now sits with the Department of Work and Pensions (DWP)."
    claim = raw["package"]["governed_claims"][0]
    claim.update(claim=headline, supporting_excerpt=headline,
                 rendered_assertion_zh_hant_hk="改革以新評核模式取代終期評核，現稱為學徒評核。")
    auxiliary = {**claim, "claim_id": "responsibility", "claim_role": "SUBSTANTIVE",
                 "claim": background, "supporting_excerpt": background,
                 "rendered_assertion_zh_hant_hk": "整體學徒計劃的責任現時由 Department of Work and Pensions（DWP）承擔。"}
    qualification = raw["package"]["qualification_evidence"][0]
    qualification["test_evidence"].update(
        material_relation_span=headline,
        new_state="end-point assessment" if fault == "old-state" else "a new assessment model, now called apprenticeship assessment",
    )
    unsupported = {**qualification, "governed_claim_id": "responsibility",
                   "test_evidence": {**qualification["test_evidence"], "change_kind": "STATUS",
                                     "material_relation_span": background,
                                     "new_state": "now sits with the Department of Work and Pensions (DWP)"}}
    raw["package"].update(governed_claims=[claim, auxiliary],
                          substantive_new_information=[headline, background],
                          qualification_evidence=[qualification, unsupported])
    if fault == "no-headline-proof":
        raw["package"]["qualification_evidence"] = [unsupported]
    body = (headline + "\n" + background).encode()
    if fault == "source-drift":
        body = headline.encode()
    acquired = SimpleNamespace(**{**vars(acquired), "body": body, "body_digest": digest_bytes(body)})
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {})
    if fault == "source-drift":
        with pytest.raises(NativeEvidenceHold, match="ASSESSOR_CLAIM_BINDING_HOLD"):
            AutonomousNativeEvidenceAssessor._validated_execution(execution, candidate, base, (source,), (acquired,))
    elif fault is not None:
        with pytest.raises(EvidencePackageError, match="qualification evidence is not exact"):
            AutonomousNativeEvidenceAssessor._validated_execution(execution, candidate, base, (source,), (acquired,))
    else:
        result = AutonomousNativeEvidenceAssessor._validated_execution(execution, candidate, base, (source,), (acquired,))
        assert len(result.governed_claims) == 2
        assert [item.governed_claim_id for item in result.qualification_evidence] == [claim["claim_id"]]
        assert [item["governed_claim_id"] for item in result.assessment_records
                if item["record_type"] == "QUALIFICATION_EVIDENCE"] == [claim["claim_id"]]


@pytest.mark.parametrize("prefix", ("", "It is false that ", "If approved; ", "Officials denied that\n"))
@pytest.mark.parametrize("object_state", (True, False))
def test_complete_replacement_object_does_not_lose_source_context(prefix, object_state):
    from newsroom.control_plane.admission import _operational_replacement_is_proven

    span = "These changes replace end-point assessment (EPA) with a new approach, called apprenticeship assessment, which allows assessment to take place throughout the apprenticeship rather than only at the end."
    state = span[span.index("a new approach"):].rstrip(".") if object_state else "end-point assessment (EPA)"
    assert _operational_replacement_is_proven(
        span, SimpleNamespace(claim=span, supporting_excerpt=span),
        source_context=prefix + span, new_state=state,
    ) is (not prefix and object_state)


def test_hko_base_upgrade_proves_original_bytes_and_rejects_other_changes(tmp_path):
    from newsroom.control_plane.native_assessor import _legacy_hko_base_digests
    from newsroom.control_plane.native_weather_evidence import hko_evidence_body
    from newsroom.tests.test_native_weather_evidence import HKO_WARNING

    connection, _port, candidate = candidate_fixture(tmp_path)
    try:
        raw = canonical_json_bytes({"WTS": HKO_WARNING})
        old = replace(_base_package(_ready_package(candidate)[1]), source_ids=("HK-02",),
                      passages=(raw.decode(),), observation_digests=(digest_bytes(raw),))
        body = hko_evidence_body(raw)
        current = replace(old, passages=(body.decode(),), observation_digests=(digest_bytes(body),))
        sources = (SimpleNamespace(unit=SimpleNamespace(source_id="HK-02")),)
        acquired = (SimpleNamespace(body=body, currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT"),)
        assert old.digest in _legacy_hko_base_digests(current, sources, acquired)
        assert _legacy_hko_base_digests(current, (), ()) == frozenset()
        assert _legacy_hko_base_digests(current, (SimpleNamespace(unit=SimpleNamespace(source_id="UK-01")),), acquired) == frozenset()
        changed = body + b" Invented information."
        assert _legacy_hko_base_digests(replace(current, passages=(changed.decode(),)), sources, (SimpleNamespace(body=changed, currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT"),)) == frozenset()
        changed_raw = canonical_json_bytes({"WTS": {**HKO_WARNING, "actionCode": "CANCEL"}})
        changed_body = hko_evidence_body(changed_raw)
        changed_base = replace(current, passages=(changed_body.decode(),),
                               observation_digests=(digest_bytes(changed_body),))
        assert old.digest not in _legacy_hko_base_digests(
            changed_base, sources,
            (SimpleNamespace(body=changed_body, currentness_basis="AUTHORITATIVE_CURRENT_CONTENT_ENDPOINT"),),
        )
        from newsroom.control_plane.native_weather_evidence import hko_completed_event_body
        historical = hko_completed_event_body(changed_raw)
        historical_base = replace(current, passages=(historical.decode(),),
                                  observation_digests=(digest_bytes(historical),))
        raw_base = replace(current, passages=(changed_raw.decode(),),
                           observation_digests=(digest_bytes(changed_raw),))
        assert _legacy_hko_base_digests(historical_base, sources, (
            SimpleNamespace(body=historical, currentness_basis="RETAINED_AUTHORITATIVE_COMPLETED_EVENT"),
        )) == frozenset({changed_base.digest, raw_base.digest})
    finally:
        connection.close()


def test_proved_hko_representation_upgrade_has_one_new_accounted_contract_attempt(tmp_path, monkeypatch):
    import newsroom.control_plane.native_assessor as module

    connection, _port, candidate = candidate_fixture(tmp_path)
    try:
        base = _base_package(_ready_package(candidate)[1])
        calls = []
        def dispatch(_prompt):
            calls.append("provider")
            return NativeAssessmentExecution(
                canonical_json_bytes({"package": _model_package_value(base)}).decode(),
                {"usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 1,
                 "cached_read_tokens": 0, "cached_write_tokens": 0, "reasoning_tokens": 0,
                 "context_tokens": 1, "total_tokens": 2},
            )
        monkeypatch.setattr(module, "VERSION", "newsroom.native-evidence-assessor.v13")
        monkeypatch.setattr(__import__(__name__, fromlist=["VERSION"]), "VERSION", module.VERSION)
        service, old_usage = _usage(tmp_path, monkeypatch)
        AutonomousNativeEvidenceAssessor(dispatch, usage=old_usage, dispatch_fence=nullcontext)(candidate, base, (), ())
        monkeypatch.setattr(module, "VERSION", "newsroom.native-evidence-assessor.v14")
        monkeypatch.setattr(__import__(__name__, fromlist=["VERSION"]), "VERSION", module.VERSION)
        _, usage = _usage(tmp_path, monkeypatch)
        current = replace(base, passages=(base.passages[0] + " Canonical field projection.",))
        # The exact byte-proof helper has its own positive/tamper test above;
        # isolate settlement/accounting behaviour here, not source acquisition.
        monkeypatch.setattr(module, "_legacy_hko_base_digests", lambda *_: frozenset({base.digest}))
        assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=usage, dispatch_fence=nullcontext)
        with pytest.raises(NativeEvidenceHold, match="INPUT_CHANGED"):
            assessor.assess_with_boundary(candidate, current, (), (), before_dispatch=None, cached_only=True)
        result = assessor(candidate, current, (), ())
        assert assessor(candidate, current, (), ()) == result
        assert assessor.assess_with_boundary(
            candidate, current, (), (), before_dispatch=None, cached_only=True,
        ) == result
        monkeypatch.setattr(module, "_legacy_hko_base_digests", lambda *_: frozenset())
        for cached_only in (True, False):
            with pytest.raises(NativeEvidenceHold, match="INPUT_CHANGED"):
                assessor.assess_with_boundary(
                    candidate, current, (), (), before_dispatch=None, cached_only=cached_only,
                )
        assert calls == ["provider", "provider"]
        with sqlite3.connect(service.path) as retained:
            assert retained.execute("SELECT count(*) FROM model_invocation_allocations").fetchone() == (2,)
            assert retained.execute("SELECT count(*) FROM model_invocation_terminals").fetchone() == (2,)
    finally:
        connection.close()


@pytest.mark.parametrize("declared", (True, False))
def test_source_declared_acronym_revalidates_retained_output_without_dispatch(
    tmp_path, monkeypatch, retained_22589_assessment, declared,
):
    from newsroom.control_plane.native_evidence import rights_eligibility_digest

    _use_historical_v16(monkeypatch)
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment,
    )
    claim_text = "Official deadline changed for GCO."
    raw_claim = raw["package"]["governed_claims"][0]
    raw_claim.update(claim=claim_text, supporting_excerpt=claim_text,
                     rendered_assertion_zh_hant_hk="GCO限期已更改。")
    raw["package"]["substantive_new_information"] = [claim_text]
    raw["package"]["qualification_evidence"][0]["test_evidence"].update(
        material_relation_span=claim_text, reader_action=claim_text,
    )
    text = ("General Consent Order (GCO).\n\n" if declared else "") + claim_text
    body = text.encode()
    base = replace(base, passages=(text,), observation_digests=(digest_bytes(body),))
    acquired = SimpleNamespace(**{
        **vars(acquired), "body": body, "body_digest": digest_bytes(body),
        "rights_eligibility_digest": rights_eligibility_digest(
            source.rights, body_digest=digest_bytes(body),
            transport_digest=acquired.transport_evidence_digest,
            exclusion_signals=(), text_only=True,
        ),
    })
    service, usage = _usage(tmp_path, monkeypatch)
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {
        "usage_basis": "PROVIDER_REPORTED", "input_tokens": 1, "output_tokens": 1,
        "cached_read_tokens": 0, "cached_write_tokens": 0,
        "reasoning_tokens": 0, "context_tokens": 1, "total_tokens": 2,
    })
    allocation = usage.begin(candidate, base, "retained declared acronym fixture")
    dispatch_at = usage.mark_dispatch(allocation)
    usage.retain_result(allocation, execution, dispatch_at=dispatch_at)
    usage.complete(allocation, outcome="ASSESSOR_VALIDATION_FAILED", execution=execution,
                   provider_dispatched=True, dispatch_at=dispatch_at,
                   failure_class="ASSESSMENT_VALIDATION_FAILED")
    def retained_rows():
        with sqlite3.connect(service.path) as connection:
            return tuple(connection.execute(f"SELECT * FROM {table}").fetchall()
                         for table in ("model_invocation_allocations", "model_invocation_terminals", "ledger"))
    before = retained_rows()
    assessor = AutonomousNativeEvidenceAssessor(
        lambda _: pytest.fail("consumer correction dispatched provider"),
        usage=usage, dispatch_fence=nullcontext,
    )
    if declared:
        result = assessor.assess_with_boundary(
            candidate, base, (source,), (acquired,), before_dispatch=None, cached_only=True,
        )
        assert result.governed_claims[0].named_entities == ("GCO",)
        assert len(result.qualification_evidence) == 1
        package = replace(base, **{
            name: getattr(result, name) for name in (
                "substantive_new_information", "governed_claims", "qualification_evidence",
                "selection_rationale", "geography", "categories", "explicit_exclusions",
            )
        })
        records = NativeEvidenceController._records(base, package, (source,), (acquired,), result)
        retained = tuple((record["record_id"], record["record_type"],
                          canonical_json_bytes(record).decode(), digest_bytes(canonical_json_bytes(record)))
                         for record in records)
        assert validate_governed_evidence_records(
            candidate_id=base.candidate_id, source_inventory=((source.unit.source_id, acquired.canonical_url),),
            base_package_digest=base.digest, package=package, retained_records=retained,
        ) is not None
    else:
        with pytest.raises(NativeEvidenceHold, match="ASSESSOR_RENDERING_CONTRACT_HOLD"):
            assessor.assess_with_boundary(
                candidate, base, (source,), (acquired,), before_dispatch=None, cached_only=True,
            )
    assert retained_rows() == before


def test_retained_process_with_separate_public_invitation_passes_static_validation(
    retained_22589_assessment,
):
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment, kind="deadline",
    )
    launch = "The changes will be subject to a consultation which was launched yesterday."
    invitation = "It seeks views from the public, industry and business to gauge how these changes would affect them."
    text = launch + " " + invitation
    claim = raw["package"]["governed_claims"][0]
    claim.update(claim=text, supporting_excerpt=text,
                 rendered_assertion_zh_hant_hk="諮詢已於昨日展開，向公眾、業界及企業徵求意見，以了解這些改動對他們的影響。")
    qualification = raw["package"]["qualification_evidence"][0]
    qualification["test_evidence"].update(
        action_class="PROCESS", material_relation_span=launch,
        reader_action="seeks views from the public, industry and business",
    )
    raw["package"]["substantive_new_information"] = [text]
    body = text.encode()
    base = replace(base, passages=(text,), observation_digests=(digest_bytes(body),))
    acquired = SimpleNamespace(**{**vars(acquired), "body": body, "body_digest": digest_bytes(body)})
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {})
    result = AutonomousNativeEvidenceAssessor._validated_execution(
        execution, candidate, base, (source,), (acquired,),
    )
    assert len(result.qualification_evidence) == 1
    assert dict(result.qualification_evidence[0].test_evidence)["reader_action"] == qualification["test_evidence"]["reader_action"]
    assert execution.text == canonical_json_bytes(raw).decode()


@pytest.mark.parametrize('available', [False, True])
def test_qualification_cached_only_has_no_judgment_or_provider_fallback(
    tmp_path, monkeypatch, retained_22589_assessment, available,
):
    from newsroom.control_plane.native_assessor_judgments import JudgedAssessment
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment, kind='policy',
    )
    raw['package'].update(substantive_new_information=[], governed_claims=[], qualification_evidence=[])
    execution=NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {})
    _,usage=_usage(tmp_path,monkeypatch)
    calls=[]
    def read(*_):
        calls.append('retained')
        return JudgedAssessment(execution,b'fixture',None)
    def forbidden(*_,**__):
        pytest.fail('cached qualification dispatched or entered fresh judgment')
    assessor=AutonomousNativeEvidenceAssessor(forbidden,usage=usage,dispatch_fence=nullcontext,
        judgments=SimpleNamespace(get_decision_ref=forbidden),qualification=forbidden,
        retained_qualification=read if available else None)
    if available:
        result=assessor.assess_with_boundary(candidate,base,(source,),(acquired,),
            before_dispatch=lambda:calls.append('consumer'),cached_only=True,qualification_cached_only=True)
        assert not result.governed_claims and calls==['consumer','retained']
    else:
        with pytest.raises(NativeEvidenceHold,match='QUALIFICATION_RETAINED_RESULT_UNAVAILABLE'):
            assessor.assess_with_boundary(candidate,base,(source,),(acquired,),
                before_dispatch=forbidden,cached_only=True,qualification_cached_only=True)
        assert calls==[]


def test_weather_record_metadata_failure_revalidates_once_per_consumer_contract():
    from newsroom.control_plane.native_assessor import assessment_revalidation_due
    from newsroom.control_plane.native_composition import ASSESSMENT_CONTRACT_VERSION

    facts = {
        "reason": "EVIDENCE_VALIDATION_HOLD",
        "assessment_contract_version": ASSESSMENT_CONTRACT_VERSION.replace(
            "newsroom.named-entity.v16", "newsroom.named-entity.v15",
        ),
    }
    assert assessment_revalidation_due(facts, ASSESSMENT_CONTRACT_VERSION)
    facts["assessment_contract_version"] = ASSESSMENT_CONTRACT_VERSION
    assert not assessment_revalidation_due(facts, ASSESSMENT_CONTRACT_VERSION)


def test_retained_elapsed_deadline_qualification_preserves_claims_without_new_model_output(
    retained_22589_assessment,
):
    # Facts and classifier are the unchanged accounted result43452. Only the
    # fixture source/candidate identity replaces live IDs; no provider executes.
    witness = json.loads((Path(__file__).parent / 'fixtures/native_assessor/deadline-43452.json').read_text())
    candidate, base, source, acquired, raw = _qualification_assessor_inputs(
        retained_22589_assessment, kind='policy',
    )
    claims = [{**item, 'source_ids': [source.unit.source_id]} for item in witness['governed_claims']]
    raw['package'].update(
        governed_claims=claims,
        substantive_new_information=[item['claim'] for item in claims],
        qualification_evidence=witness['qualification_evidence'],
    )
    body = witness['source_text'].encode()
    acquired = SimpleNamespace(**{**vars(acquired), 'body': body, 'body_digest': digest_bytes(body)})
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {})
    result = AutonomousNativeEvidenceAssessor._validated_execution(
        execution, candidate, base, (source,), (acquired,),
    )
    assert [claim.claim for claim in result.governed_claims] == [item['claim'] for item in claims]
    assert [item.governed_claim_id for item in result.qualification_evidence] == ['c1']
    assert dict(result.qualification_evidence[0].test_evidence)['change_kind'] == 'OFFICIAL_DEADLINE'


def test_literal_csv_names_survive_full_assessment_and_governed_records(retained_22589_assessment):
    """Synthetic evidence proves representation, not a production news decision."""
    import csv
    import io
    from newsroom.control_plane.govuk_spreadsheet import _Text, _csv
    from newsroom.control_plane.native_evidence import rights_eligibility_digest

    candidate, base, source, acquired, raw = _qualification_assessor_inputs(retained_22589_assessment)
    title = 'The department announced the application deadline changed to 30 June 2026'
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(["Senior Official's Name ", 'Date ', 'Name of individual or organisation ', 'Purpose of Meeting'])
    writer.writerow(['Marianthi Leontaridi', '2026-05-21', 'Boston Consulting Group', title])
    output = _Text()
    _csv(buffer.getvalue().encode(), output)
    row = output.lines[-1]
    body = title + '\n\nAttachment: https://assets.publishing.service.gov.uk/media/fixture/meetings.csv\n' + '\n'.join(output.lines)
    encoded = body.encode()
    base = replace(base, passages=(body,), observation_digests=(digest_bytes(encoded),))
    acquired = SimpleNamespace(**{
        **vars(acquired), 'body': encoded, 'body_digest': digest_bytes(encoded),
        'rights_eligibility_digest': rights_eligibility_digest(
            source.rights, body_digest=digest_bytes(encoded), transport_digest=acquired.transport_evidence_digest,
            exclusion_signals=(), text_only=True,
        ),
    })
    template = raw['package']['governed_claims'][0]
    headline = {**template, 'claim_id':'csv-headline', 'claim':title, 'supporting_excerpt':title,
                'claim_role':'HEADLINE', 'rendered_assertion_zh_hant_hk':'部門宣布申請限期改為2026年6月30日。',
                'localised_factual_expressions':[['30 June 2026','2026年6月30日']], 'quotations':[]}
    data_claim = {**headline, 'claim_id':'csv-row', 'claim':row, 'supporting_excerpt':row,
                  'claim_role':'SUBSTANTIVE', 'rendered_assertion_zh_hant_hk':
                  '第2行紀錄：Marianthi Leontaridi，日期2026-05-21，Boston Consulting Group；部門宣布申請限期改為2026年6月30日。'}
    witnesses = {'action_class':'OFFICIAL_DEADLINE','event_polarity':'AFFIRMED',
                 'action_relation':'NEW_OR_CHANGED_OFFICIAL_ACTION',
                 'material_relation_span':title,'reader_action':title}
    raw['package'].update(substantive_new_information=[title,row], governed_claims=[headline,data_claim],
                          qualification_evidence=[{'test':'OFFICIAL_ACTION_OR_DEADLINE', 'governed_claim_id':'csv-headline',
                                                   'test_evidence':witnesses, 'policy_version':'newsroom.evid-012.v7'}])
    execution = NativeAssessmentExecution(canonical_json_bytes(raw).decode(), {})
    assessment = AutonomousNativeEvidenceAssessor._validated_execution(execution, candidate, base, (source,), (acquired,))
    assert assessment.governed_claims[1].named_entities == ('Boston Consulting Group','Marianthi Leontaridi')
    governed = replace(base, substantive_new_information=assessment.substantive_new_information,
                       governed_claims=assessment.governed_claims, qualification_evidence=assessment.qualification_evidence,
                       selection_rationale=assessment.selection_rationale, geography=assessment.geography,
                       categories=assessment.categories, explicit_exclusions=assessment.explicit_exclusions)
    records = NativeEvidenceController._records(base, governed, (source,), (acquired,), assessment)
    retained = tuple((r['record_id'],r['record_type'],canonical_json_bytes(r).decode(),digest_bytes(canonical_json_bytes(r))) for r in records)
    assert validate_governed_evidence_records(candidate_id=candidate.candidate_id,
        source_inventory=((source.unit.source_id, acquired.canonical_url),), base_package_digest=base.digest,
        package=governed, retained_records=retained)
    from newsroom.control_plane.admission import _qualification_relation_is_proven
    assert _qualification_relation_is_proven(governed.qualification_evidence[0],governed.governed_claims[0],source_context=body)
    # A copied name does not authorise a paraphrased claim or a new country.
    for mutation in ('paraphrase','new_entity','changed_row'):
        altered = json.loads(canonical_json_bytes(raw))
        claim = altered['package']['governed_claims'][1]
        if mutation == 'paraphrase':claim['claim'] = 'Marianthi Leontaridi met Boston Consulting Group.'
        elif mutation == 'new_entity':claim['rendered_assertion_zh_hant_hk'] += ' UK'
        else:claim['claim'] = claim['claim'].replace('Row 2:', 'Row 3:')
        with pytest.raises((NativeEvidenceHold, EvidencePackageError)):
            AutonomousNativeEvidenceAssessor._validated_execution(NativeAssessmentExecution(canonical_json_bytes(altered).decode(), {}),candidate,base,(source,),(acquired,))


def test_current_profile_preserves_v15_v16_v17_v18_v19_v20_contracts():
    from newsroom.control_plane.native_assessor import (
        _V15_SYSTEM, _V15_SCHEMA_DIGEST, _V15_SCHEMA_BYTES, _V16_SYSTEM,
        _V17_SYSTEM, _V17_PROVIDER_SCHEMA_DIGEST,
        _V18_SYSTEM, _V18_PROVIDER_SCHEMA,
        _V19_SYSTEM, _V19_PROVIDER_SCHEMA, _V19_PROVIDER_SCHEMA_DIGEST,
        _V20_SYSTEM, _V20_PROVIDER_SCHEMA, _V20_PROVIDER_SCHEMA_DIGEST,
    )
    assert VERSION == 'newsroom.native-evidence-assessor.v23'
    assert digest_bytes(_V15_SYSTEM.encode()) == 'sha256:5788c3e827199e12932d106ad494c80b71b2691e3f9c7e535a44c5d09811d4a6'
    assert len(_V15_SYSTEM.encode()) == 6797
    assert SCHEMA_DIGEST == _V15_SCHEMA_DIGEST
    assert len(canonical_json_bytes(SCHEMA)) == _V15_SCHEMA_BYTES
    assert _V16_SYSTEM.startswith(_V15_SYSTEM)
    assert 'Illustration only, not evidence for the current candidate:' in _V16_SYSTEM
    assert 'same complete literal Row line' in _V16_SYSTEM
    assert 'CSV JSON delimiter quotes are not attributed speech' in _V16_SYSTEM
    assert 'existing empty governed_claims/qualification_evidence/no-new-information path' in _V16_SYSTEM
    assert 'claim_range and support_range' in _V17_SYSTEM
    assert 'N+1 rendered_fragments' in _V17_SYSTEM
    assert 'source_range' in SYSTEM
    assert 'rendered_assertion_zh_hant_hk_fragments' in SYSTEM
    assert _V17_PROVIDER_SCHEMA_DIGEST != PROVIDER_SCHEMA_DIGEST
    assert _V20_SYSTEM == _V19_SYSTEM == _V18_SYSTEM
    assert digest_bytes(_V20_SYSTEM.encode()) == 'sha256:9c9971f005ef0bc6450585d1e1afb440adf90576bb59f68594e464ae318c84f5'
    assert len(_V20_SYSTEM.encode()) == 5024
    assert _V20_PROVIDER_SCHEMA == _V19_PROVIDER_SCHEMA == _V18_PROVIDER_SCHEMA
    assert _V20_PROVIDER_SCHEMA_DIGEST == _V19_PROVIDER_SCHEMA_DIGEST == 'sha256:ee75aaced2b407df8051fd508af1626bedefca8c9e900af65320bba7aff1c1a0'
    assert SYSTEM != _V20_SYSTEM
    assert PROVIDER_SCHEMA_DIGEST != _V20_PROVIDER_SCHEMA_DIGEST
    assert 'substantive_claim_indexes' not in SYSTEM
    assert PROVIDER_SCHEMA['properties']['package']['properties']['select_new_information'] == {'type':'boolean'}
    assert native_assessor_module.MODEL == 'grok-4.7'
    assert native_assessor_module.REASONING == 'high'
    assert native_assessor_module.COMMAND_FLAGS != CONT_PRIMARY_COMMAND_FLAGS
    assert PROVIDER_SCHEMA_DIGEST != SCHEMA_DIGEST
    assert PROVIDER_SCHEMA['properties']['package']['properties'][
        'governed_claims'
    ]['items']['properties'].get('claim') is None


def test_wire_only_v20_to_v21_transition_never_requests_historical_reassessment():
    from newsroom.control_plane.native_assessor import assessment_revalidation_due
    facts = {'assessment_contract_version':'newsroom.native-evidence-assessor.v20+consumer.v1',
             'reason':'ASSESSOR_QUALIFICATION_CONTRACT_HOLD'}
    assert assessment_revalidation_due(facts,'newsroom.native-evidence-assessor.v21+consumer.v1') is False
    # A separate consumer-contract change keeps its existing revalidation semantics.
    assert assessment_revalidation_due(facts,'newsroom.native-evidence-assessor.v21+consumer.v2') is True


def test_frozen_v20_retained_result_replays_under_v21_without_redispatch(tmp_path, monkeypatch):
    module = native_assessor_module
    connection,_port,candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    legacy_wire = _empty_reference_result()
    legacy_wire['package'].pop('select_new_information')
    legacy_wire['package']['substantive_claim_indexes'] = []
    execution = NativeAssessmentExecution(canonical_json_bytes(legacy_wire).decode(), {
        'usage_basis':'PROVIDER_REPORTED','input_tokens':1,'output_tokens':1,
        'cached_read_tokens':0,'cached_write_tokens':0,'reasoning_tokens':0,
        'context_tokens':1,'total_tokens':2,
    })
    with monkeypatch.context() as previous:
        previous.setattr(module,'VERSION',module._V20_PRODUCER_VERSION)
        previous.setattr(module,'SYSTEM',module._V20_SYSTEM)
        previous.setattr(module,'PROVIDER_SCHEMA',module._V20_PROVIDER_SCHEMA)
        previous.setattr(module,'PROVIDER_SCHEMA_DIGEST',module._V20_PROVIDER_SCHEMA_DIGEST)
        service,usage = _usage(tmp_path,previous)
        result = AutonomousNativeEvidenceAssessor(lambda _prompt:execution,
            usage=usage,dispatch_fence=nullcontext)(candidate,base,(),())
    _service,current_usage = _usage(tmp_path,monkeypatch)
    def snapshot():
        with sqlite3.connect(service.path) as retained:
            return tuple(tuple(retained.execute(f'SELECT * FROM {table}').fetchall())
                         for table in ('model_invocation_allocations','model_invocation_terminals','ledger'))
    original = snapshot()
    def dispatch(_prompt):
        pytest.fail('historical result dispatched a v21 provider leaf')
    assessor = AutonomousNativeEvidenceAssessor(dispatch,usage=current_usage,dispatch_fence=nullcontext)
    for _ in range(2):
        assert assessor.assess_with_boundary(candidate,base,(),(),before_dispatch=None,cached_only=True) == result
    assert snapshot() == original
    connection.close()


def test_v21_input_bound_remains_exact_after_producer_only_prompt_change(tmp_path, monkeypatch):
    _service, usage = _usage(tmp_path, monkeypatch)
    historical = replace(usage._policy,
        prompt_contract_version='newsroom.native-evidence-assessor.v21')
    bound = native_assessment_input_bound(historical)
    assert bound['system_digest'] == 'sha256:e8a8f919488537711b420c6e4ddee1e7bd88adc4b5334d31f514377887ebe2d8'
    assert bound['system_bytes'] == 5_018
    assert bound['schema_digest'] == 'sha256:19a551165b34c6c8da5fc7332b4c60122e6dda773626d59d85491f76c6d79903'
    assert bound['max_request_bytes'] == 61_726
    assert native_assessment_input_bound(usage._policy)['max_request_bytes'] < 61_726


@pytest.mark.parametrize('previous', [
    'newsroom.native-evidence-assessor.v20', 'newsroom.native-evidence-assessor.v21',
])
def test_status_prompt_change_does_not_schedule_historical_reassessment(previous):
    from newsroom.control_plane.native_assessor import assessment_revalidation_due
    facts = {'assessment_contract_version': previous + '+consumer.v1',
             'reason': 'ASSESSOR_QUALIFICATION_CONTRACT_HOLD'}
    assert not assessment_revalidation_due(facts, VERSION + '+consumer.v1')
    assert assessment_revalidation_due(facts, VERSION + '+consumer.v2')
    facts['reason'] = 'NO_QUALIFYING_NEW_INFORMATION'
    assert not assessment_revalidation_due(facts, VERSION + '+consumer.v1')


def _old_provider_failure(tmp_path, monkeypatch, *, contract='newsroom.native-evidence-assessor.v21'):
    from dataclasses import asdict
    from newsroom.control_plane.writer import CliTimeoutError
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    current_system = native_assessor_module.SYSTEM
    monkeypatch.setattr(native_assessor_module, 'VERSION', contract)
    monkeypatch.setattr(native_assessor_module, 'SYSTEM', native_assessor_module._V21_SYSTEM)
    service, old = _usage(tmp_path, monkeypatch)
    policy = InvocationEfficiencyPolicy.create(**{
        **asdict(old._policy), 'hard_estimate_ceiling_tokens': 300_000,
    })
    old = NativeAssessmentUsage(service, policy, clock=old._clock)
    def timeout(_request): raise CliTimeoutError('fixture writer timed out')
    with pytest.raises(CliTimeoutError):
        AutonomousNativeEvidenceAssessor(timeout, usage=old, dispatch_fence=nullcontext)(candidate, base, (), ())
    with sqlite3.connect(service.path) as retained:
        allocation = retained.execute('SELECT record_json FROM model_invocation_allocations').fetchone()[0]
        terminal = retained.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0]
    monkeypatch.setattr(native_assessor_module, 'VERSION', native_assessor_module._REFERENCE_PRODUCER_VERSION)
    monkeypatch.setattr(native_assessor_module, 'SYSTEM', current_system)
    _, current = _usage(tmp_path, monkeypatch)
    return connection, candidate, base, service, current, allocation, terminal


def test_known_old_provider_failure_allows_one_current_producer_and_preserves_unknown_accounting(tmp_path, monkeypatch):
    connection, candidate, base, service, current, old_allocation, old_terminal = _old_provider_failure(tmp_path, monkeypatch)
    calls = []
    execution = NativeAssessmentExecution(json.dumps(_empty_reference_result()), {
        'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 1, 'output_tokens': 1,
        'cached_read_tokens': 0, 'cached_write_tokens': 0, 'reasoning_tokens': 0,
        'context_tokens': 1, 'total_tokens': 2,
    })
    def dispatch(_request): calls.append('current'); return execution
    assessor = AutonomousNativeEvidenceAssessor(dispatch, usage=current, dispatch_fence=nullcontext)
    try:
        first = assessor(candidate, base, (), ())
        reopened = NativeAssessmentUsage(ModelUsageService(service.path), current._policy, clock=current._clock)
        assert AutonomousNativeEvidenceAssessor(dispatch, usage=reopened, dispatch_fence=nullcontext)(candidate, base, (), ()) == first
        assert calls == ['current']
        assert current.retained_output_contract_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT COUNT(*) FROM model_invocation_allocations').fetchone() == (2,)
            assert retained.execute('SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?',
                (json.loads(old_allocation)['invocation_id'],)).fetchone() == (old_allocation,)
            assert retained.execute('SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?',
                (json.loads(old_terminal)['invocation_id'],)).fetchone() == (old_terminal,)
        assert json.loads(old_terminal)['components']['total_tokens'] == 300_000
        assert json.loads(old_terminal)['usage_status'] == 'ESTIMATED'
        assert json.loads(old_terminal)['pre_dispatch_zero_proved'] is False
    finally:
        connection.close()


@pytest.mark.parametrize('defect', [
    'active', 'current-producer', 'unknown-producer', 'unqualified-current',
    'allocation-index', 'terminal-index', 'candidate-binding', 'source-binding',
    'upper-charge', 'retained-output', 'telemetry', 'cached-only', 'changed-source',
])
def test_old_provider_failure_continuation_denies_incomplete_or_changed_authority(tmp_path, monkeypatch, defect):
    from dataclasses import asdict
    contract = {'current-producer': native_assessor_module._REFERENCE_PRODUCER_VERSION,
                'unknown-producer': 'newsroom.native-evidence-assessor.v999'}.get(defect,
                    'newsroom.native-evidence-assessor.v21')
    connection, candidate, base, service, current, old_allocation, old_terminal = _old_provider_failure(
        tmp_path, monkeypatch, contract=contract)
    invocation_id = json.loads(old_allocation)['invocation_id']
    with sqlite3.connect(service.path) as retained:
        if defect == 'active': retained.execute('DELETE FROM model_invocation_terminals')
        elif defect == 'allocation-index': retained.execute("UPDATE model_invocation_allocations SET request_digest='sha256:'||?", ('f' * 64,))
        elif defect == 'terminal-index': retained.execute("UPDATE model_invocation_terminals SET failure_class='wrong'")
        elif defect == 'candidate-binding':
            raw = json.loads(retained.execute('SELECT record_json FROM model_work_envelopes').fetchone()[0])
            raw['candidate_id'] = 'unknown-candidate'
            retained.execute('UPDATE model_work_envelopes SET record_json=?', (canonical_json_bytes(raw).decode(),))
        elif defect == 'source-binding':
            raw = json.loads(retained.execute('SELECT record_json FROM model_invocation_context_manifests').fetchone()[0])
            raw['source_reference_binding']['partition_version'] = 'source-view.v999'
            retained.execute('UPDATE model_invocation_context_manifests SET record_json=?', (canonical_json_bytes(raw).decode(),))
        elif defect == 'upper-charge':
            from newsroom.control_plane.model_usage import _terminal_from_record, InvocationTerminal
            terminal = _terminal_from_record(json.loads(old_terminal))
            values = {name: getattr(terminal, name) for name in terminal.__dataclass_fields__}
            terminal = InvocationTerminal.create(**{**values, 'components': replace(terminal.components, total_tokens=299_999)})
            retained.execute('UPDATE model_invocation_terminals SET terminal_digest=?,record_json=?',
                (terminal.terminal_digest, canonical_json_bytes(terminal.as_record()).decode()))
        elif defect == 'retained-output':
            from newsroom.control_plane.store import append_ledger
            append_ledger(retained, 'NATIVE_ASSESSMENT_RESULT', {'invocation_id': invocation_id})
        elif defect == 'telemetry':
            retained.execute('INSERT INTO model_provider_telemetry VALUES(?,?,?,?)',
                ('sha256:' + 'b' * 64, invocation_id, 'sha256:' + 'c' * 64, '{}'))
    if defect == 'unqualified-current':
        current._policy = InvocationEfficiencyPolicy.create(**{**asdict(current._policy), 'qualified': False})
    if defect == 'changed-source': base = replace(base, passages=(base.passages[0] + ' changed',))
    calls = []
    assessor = AutonomousNativeEvidenceAssessor(lambda _request: calls.append('provider'), usage=current, dispatch_fence=nullcontext)
    try:
        with pytest.raises(NativeEvidenceHold):
            assessor.assess_with_boundary(candidate, base, (), (), before_dispatch=None, cached_only=defect == 'cached-only')
        assert calls == []
        assert current.retained_output_contract_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT COUNT(*) FROM model_invocation_allocations').fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize('boundary', ['stop', 'currentness'])
def test_old_provider_failure_continuation_keeps_current_source_and_stop_boundaries(tmp_path, monkeypatch, boundary):
    from newsroom.control_plane.veto import VetoError
    connection, candidate, base, service, current, _allocation, _terminal = _old_provider_failure(tmp_path, monkeypatch)
    calls = []
    assessor = AutonomousNativeEvidenceAssessor(lambda _request: calls.append('provider'), usage=current, dispatch_fence=nullcontext)
    def stop(): raise VetoError('owner stop')
    sources, acquired = (), ()
    if boundary == 'currentness':
        sources = (SimpleNamespace(unit=SimpleNamespace(source_id=base.source_ids[0])),)
        acquired = (SimpleNamespace(currentness_basis='STALE_VERSION'),)
    try:
        with pytest.raises(VetoError if boundary == 'stop' else NativeEvidenceHold):
            assessor.assess_with_boundary(candidate, base, sources, acquired,
                before_dispatch=stop if boundary == 'stop' else None, cached_only=False)
        assert calls == []
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT COUNT(*) FROM model_invocation_allocations').fetchone() == (2 if boundary == 'stop' else 1,)
            assert retained.execute('SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?',
                (json.loads(_terminal)['invocation_id'],)).fetchone() == (_terminal,)
    finally:
        connection.close()


def test_unknown_current_producer_can_be_read_as_separate_semantic_origin_not_retry(tmp_path, monkeypatch):
    # Existing failure fixture deliberately uses v21 prompt bytes. A current
    # origin must instead bind the actual current system contract.
    monkeypatch.setattr(native_assessor_module, '_V21_SYSTEM', native_assessor_module.SYSTEM)
    connection, candidate, base, service, current, old_allocation, old_terminal = _old_provider_failure(
        tmp_path, monkeypatch, contract=native_assessor_module._REFERENCE_PRODUCER_VERSION)
    try:
        assert current.retained_assessments(candidate, base) is None
        origin = current.retained_semantic_origin_failure(candidate)
        assert origin is not None and origin.execution is None
        assert origin.contract_version == native_assessor_module.VERSION
        assert origin.proof.invocation_id == json.loads(old_allocation)['invocation_id']
        assert current.retained_assessments(candidate, base) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT record_json FROM model_invocation_allocations').fetchone()[0] == old_allocation
            assert retained.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0] == old_terminal
    finally:
        connection.close()


def _reported_validation_origin(tmp_path, monkeypatch, contract):
    """Genuine disposable accounting; returned bytes are intentionally invalid."""
    connection, _port, candidate = candidate_fixture(tmp_path)
    base = _base_package(_ready_package(candidate)[1])
    with monkeypatch.context() as historical:
        historical.setattr(native_assessor_module, 'VERSION', contract)
        historical.setattr(native_assessor_module, 'SYSTEM', getattr(
            native_assessor_module, '_V15_SYSTEM' if contract.endswith('.v15') else '_V17_SYSTEM'))
        historical.setattr(native_assessor_module, 'PROVIDER_SCHEMA_DIGEST',
            SCHEMA_DIGEST if contract.endswith('.v15') else native_assessor_module._V17_PROVIDER_SCHEMA_DIGEST)
        service, usage = _usage(tmp_path, historical)
        allocation = usage.begin(candidate, base, 'Synthetic retained validation-origin fixture')
        dispatch_at = usage.mark_dispatch(allocation)
        execution = NativeAssessmentExecution('{', {
            'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 1, 'output_tokens': 1,
            'cached_read_tokens': 0, 'cached_write_tokens': 0, 'reasoning_tokens': 0,
            'context_tokens': 1, 'total_tokens': 2,
        })
        usage.retain_result(allocation, execution, dispatch_at=dispatch_at)
        usage.complete(allocation, outcome='ASSESSOR_VALIDATION_FAILED', execution=execution,
            provider_dispatched=True, dispatch_at=dispatch_at, failure_class='ASSESSMENT_VALIDATION_FAILED')
    _, current = _usage(tmp_path, monkeypatch)
    return connection, candidate, service, current, allocation


@pytest.mark.parametrize('contract', ['newsroom.native-evidence-assessor.v15', 'newsroom.native-evidence-assessor.v17'])
def test_reported_validation_is_authenticated_semantic_origin_not_executable_copy(tmp_path, monkeypatch, contract):
    connection, candidate, service, current, allocation = _reported_validation_origin(tmp_path, monkeypatch, contract)
    with sqlite3.connect(service.path) as retained:
        before = tuple(retained.execute(f'SELECT * FROM {table}').fetchall() for table in (
            'model_invocation_allocations', 'model_invocation_terminals', 'ledger'))
    try:
        for _ in range(2):
            origin = current.retained_semantic_origin_failure(candidate)
            assert origin is not None and origin.outcome == 'ASSESSOR_VALIDATION_FAILED'
            assert origin.contract_version == contract
            assert origin.proof.invocation_id == allocation.invocation_id
            assert origin.result_digest == digest_bytes(b'{')
            assert origin.result_receipt_digest.startswith('sha256:')
            assert current.retained_old_provider_failure(candidate) is None
            if contract.endswith('.v17'):
                assert origin.execution is None  # Missing materialisation is not a copy grant.
        with sqlite3.connect(service.path) as retained:
            after = tuple(retained.execute(f'SELECT * FROM {table}').fetchall() for table in (
                'model_invocation_allocations', 'model_invocation_terminals', 'ledger'))
        assert after == before
    finally:
        connection.close()


@pytest.mark.parametrize('defect', ['foreign-candidate', 'raw-tamper', 'missing-raw', 'partial-manifest',
    'active', 'unreported', 'telemetry', 'allocation-binding', 'result-retired', 'unqualified-current'])
def test_reported_validation_semantic_origin_denies_incomplete_authenticated_footprint(tmp_path, monkeypatch, defect):
    from dataclasses import asdict
    connection, candidate, service, current, allocation = _reported_validation_origin(
        tmp_path, monkeypatch, 'newsroom.native-evidence-assessor.v17')
    with sqlite3.connect(service.path) as retained:
        if defect == 'raw-tamper':
            raw = json.loads(retained.execute("SELECT payload_json FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'").fetchone()[0])
            raw['result_text'] = 'altered'
            retained.execute("UPDATE ledger SET payload_json=? WHERE kind='NATIVE_ASSESSMENT_RESULT'", (canonical_json_bytes(raw).decode(),))
        elif defect in {'missing-raw', 'result-retired'}:
            retained.execute("DELETE FROM ledger WHERE kind='NATIVE_ASSESSMENT_RESULT'")
        elif defect == 'partial-manifest':
            retained.execute('DELETE FROM model_invocation_context_manifests')
        elif defect == 'active':
            retained.execute('DELETE FROM model_invocation_terminals')
        elif defect == 'unreported':
            retained.execute("UPDATE model_invocation_terminals SET usage_status='ESTIMATED'")
        elif defect == 'telemetry':
            retained.execute('DELETE FROM model_provider_telemetry')
        elif defect == 'allocation-binding':
            retained.execute("UPDATE model_invocation_allocations SET request_digest='sha256:'||?", ('f' * 64,))
    if defect == 'foreign-candidate': candidate = replace(candidate, candidate_id='11111111-1111-4111-8111-111111111111')
    if defect == 'unqualified-current':
        current._policy = InvocationEfficiencyPolicy.create(**{**asdict(current._policy), 'qualified': False})
    try:
        assert current.retained_semantic_origin_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT count(*) FROM model_invocation_allocations').fetchone() == (1,)
    finally:
        connection.close()


def _unallocated_semantic_origin_envelope(service, candidate, *, contract='newsroom.native-evidence-assessor.v19',
    candidate_id=None, hypothesis_digest=None, cycle_id=None, evidence_package_digest=None):
    with sqlite3.connect(service.path) as retained:
        base_digest = json.loads(retained.execute('SELECT record_json FROM model_work_envelopes LIMIT 1').fetchone()[0])['evidence_package_digest']
    base_digest = evidence_package_digest or base_digest
    envelope = WorkEnvelope.create(
        cycle_id=cycle_id or native_assessor_module._assessment_cycle_id(candidate.version_id, base_digest, contract),
        workload_class=WorkloadClass.NATIVE_EVIDENCE_ASSESSOR,
        admitted_at=datetime(2026, 9, 9, tzinfo=UTC), admission_decision_id=None,
        candidate_id=candidate_id or candidate.candidate_id,
        hypothesis_digest=hypothesis_digest or candidate.governing_manifest.canonical_digest,
        evidence_package_digest=base_digest, ingest_id=None, graphiti_attempt_id=None)
    service.open_envelope(envelope)
    return envelope


@pytest.mark.parametrize('contract', ['newsroom.native-evidence-assessor.v15', 'newsroom.native-evidence-assessor.v17'])
def test_semantic_origin_keeps_later_unallocated_native_footprint_unresolved(tmp_path, monkeypatch, contract):
    connection, candidate, service, current, allocation = _reported_validation_origin(tmp_path, monkeypatch, contract)
    envelope = _unallocated_semantic_origin_envelope(service, candidate)
    def rows():
        with sqlite3.connect(service.path) as retained:
            return tuple(retained.execute(f'SELECT * FROM {table}').fetchall() for table in (
                'model_work_envelopes', 'model_invocation_allocations', 'model_invocation_terminals', 'ledger'))
    before = rows()
    try:
        assert current.retained_assessments(candidate) is None  # Ordinary base guard is unchanged.
        for _ in range(2):
            origin = current.retained_semantic_origin_failure(candidate)
            assert origin is not None and origin.outcome == 'ASSESSOR_VALIDATION_FAILED'
            assert origin.proof.invocation_id == allocation.invocation_id
            assert origin.result_digest == digest_bytes(b'{')
            assert origin.proof.envelope_id != envelope.envelope_id
        assert current.retained_assessments(candidate) is None
        assert current.retained_pre_dispatch_failure(candidate) is None  # Old paid call is never zero.
        assert rows() == before
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT count(*) FROM model_invocation_allocations WHERE envelope_id=?',
                (envelope.envelope_id,)).fetchone() == (0,)
    finally:
        connection.close()


def _secondary_semantic_origin_allocation(tmp_path, monkeypatch, candidate, *, outcome):
    base = _base_package(_ready_package(candidate)[1])
    with monkeypatch.context() as historical:
        historical.setattr(native_assessor_module, 'VERSION', native_assessor_module._V21_PRODUCER_VERSION)
        historical.setattr(native_assessor_module, 'SYSTEM', native_assessor_module._V21_SYSTEM)
        service, usage = _usage(tmp_path, historical)
        usage._clock = lambda: datetime(2026, 9, 10, tzinfo=UTC)
        allocation = usage.begin(candidate, base, 'Synthetic later distinct paid-purpose fixture')
        if outcome == 'active': return allocation
        dispatch_at = usage.mark_dispatch(allocation)
        execution = None if outcome == 'unknown' else NativeAssessmentExecution('{', {} if outcome == 'unreported' else {
            'usage_basis': 'PROVIDER_REPORTED', 'input_tokens': 1, 'output_tokens': 1,
            'cached_read_tokens': 0, 'cached_write_tokens': 0, 'reasoning_tokens': 0,
            'context_tokens': 1, 'total_tokens': 2,
        })
        if execution is not None: usage.retain_result(allocation, execution, dispatch_at=dispatch_at)
        usage.complete(allocation, outcome='ASSESSOR_PROVIDER_FAILED' if outcome == 'unknown' else 'ASSESSOR_VALIDATION_FAILED',
            execution=execution, provider_dispatched=True, dispatch_at=dispatch_at,
            failure_class='UNKNOWN_PROVIDER_FAILURE' if outcome == 'unknown' else 'ASSESSMENT_VALIDATION_FAILED')
    return allocation


@pytest.mark.parametrize('defect', ['wrong-cycle', 'wrong-candidate', 'wrong-hypothesis', 'wrong-base',
    'unknown-producer', 'row-tamper', 'active', 'unknown', 'unreported', 'deleted-allocation-anchor'])
def test_semantic_origin_unallocated_boundary_denies_changed_or_unsettled_footprints(tmp_path, monkeypatch, defect):
    connection, candidate, service, current, old_allocation = _reported_validation_origin(
        tmp_path, monkeypatch, 'newsroom.native-evidence-assessor.v17')
    kwargs = {}
    if defect == 'wrong-cycle': kwargs['cycle_id'] = 'foreign-cycle'
    elif defect == 'wrong-hypothesis': kwargs['hypothesis_digest'] = 'sha256:' + 'f' * 64
    elif defect == 'unknown-producer': kwargs['contract'] = 'newsroom.native-evidence-assessor.v999'
    elif defect == 'wrong-base':
        with sqlite3.connect(service.path) as retained:
            base = json.loads(retained.execute('SELECT record_json FROM model_work_envelopes LIMIT 1').fetchone()[0])['evidence_package_digest']
        kwargs.update(cycle_id=native_assessor_module._assessment_cycle_id(candidate.version_id, base,
            'newsroom.native-evidence-assessor.v19'), evidence_package_digest='sha256:' + 'f' * 64)
    envelope = _unallocated_semantic_origin_envelope(service, candidate, **kwargs)
    if defect in {'active', 'unknown', 'unreported', 'deleted-allocation-anchor'}:
        allocation = _secondary_semantic_origin_allocation(tmp_path, monkeypatch, candidate,
            outcome='reported' if defect == 'deleted-allocation-anchor' else defect)
        if defect == 'deleted-allocation-anchor':
            with sqlite3.connect(service.path) as retained:
                retained.execute('PRAGMA foreign_keys=OFF')
                retained.execute('DELETE FROM model_invocation_allocations WHERE invocation_id=?', (allocation.invocation_id,))
                assert retained.execute('SELECT 1 FROM model_invocation_terminals WHERE invocation_id=?',
                    (allocation.invocation_id,)).fetchone() is not None
    elif defect == 'row-tamper':
        with sqlite3.connect(service.path) as retained:
            raw = envelope.as_record(); raw['cycle_id'] = 'altered'
            retained.execute('UPDATE model_work_envelopes SET record_json=? WHERE envelope_id=?',
                (canonical_json_bytes(raw).decode(), envelope.envelope_id))
    elif defect == 'wrong-candidate':
        candidate = replace(candidate, candidate_id='11111111-1111-4111-8111-111111111111')
    try:
        assert current.retained_semantic_origin_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT record_json FROM model_invocation_allocations WHERE invocation_id=?',
                (old_allocation.invocation_id,)).fetchone() == (canonical_json_bytes(old_allocation.as_record()).decode(),)
    finally:
        connection.close()


def test_pending_native_envelope_does_not_expand_unknown_semantic_origin_eligibility(tmp_path, monkeypatch):
    connection, candidate, base, service, current, allocation, terminal = _old_provider_failure(tmp_path, monkeypatch)
    try:
        assert current.retained_semantic_origin_failure(candidate).outcome == 'ASSESSOR_PROVIDER_FAILED'
        _unallocated_semantic_origin_envelope(service, candidate)
        assert current.retained_semantic_origin_failure(candidate) is None
        with sqlite3.connect(service.path) as retained:
            assert retained.execute('SELECT record_json FROM model_invocation_terminals').fetchone()[0] == terminal
    finally:
        connection.close()
