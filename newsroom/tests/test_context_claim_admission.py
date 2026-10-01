from dataclasses import replace

import pytest

from newsroom.control_plane.admission import DeterministicWriteAdmission
from newsroom.control_plane.editorial import StoryCandidateRecord
from newsroom.control_plane.evidence import EvidencePackage, GovernedClaimStatus
from newsroom.control_plane.writer import (
    FixtureWriter,
    WriterCopy,
    _prompt,
    _writer_evidence_value,
    required_surface_copy,
    validate_writer_copy,
)
from newsroom.tests.test_zero_quota_write_loop import _bind_fixture_entities, _candidate_package


_CONTEXT_STATUSES = (
    GovernedClaimStatus.EXPRESSLY_PROVISIONAL_FACT,
    GovernedClaimStatus.CONTEXTUAL_BACKGROUND,
)


def _context_package(status: GovernedClaimStatus) -> tuple[StoryCandidateRecord, EvidencePackage]:
    candidate, package = _candidate_package()
    context = replace(
        package.governed_claims[1],
        claim_id="fixture:context",
        claim="The cost estimate remains provisional.",
        supporting_excerpt="The cost estimate remains provisional.",
        claim_role="CONTEXT",
        status=status,
        rendered_assertion_zh_hant_hk="費用估算仍屬暫定。",
        semantic_relation_evidence_id="fixture:context:semantic",
    )
    claims = (*package.governed_claims, context)
    return candidate, replace(
        package,
        passages=(package.passages[0] + "\n" + context.claim,),
        governed_claims=claims,
        evidence_gate_evidence=tuple(
            replace(gate, governed_claim_ids=tuple(claim.claim_id for claim in claims))
            for gate in package.evidence_gate_evidence
        ),
        resolved_evidence_records=(
            *package.resolved_evidence_records,
            (context.semantic_relation_evidence_id, "fixture:context:digest"),
        ),
    )


@pytest.mark.parametrize("status", (GovernedClaimStatus.CONFIRMED_FACT, *_CONTEXT_STATUSES))
def test_qualified_confirmed_core_allows_faithful_context(status: GovernedClaimStatus) -> None:
    candidate, package = _context_package(status)

    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-10-01T00:00:00Z"
    )

    assert decision.decision == "WRITE_READY"
    assert decision.stable_reason_codes == ("QUALIFIED_WRITE_READY",)


@pytest.mark.parametrize("status", _CONTEXT_STATUSES)
@pytest.mark.parametrize("role", ("HEADLINE", "SUBSTANTIVE"))
def test_non_confirmed_core_claim_stays_held(status: GovernedClaimStatus, role: str) -> None:
    candidate, package = _candidate_package()
    package = replace(
        package,
        governed_claims=tuple(
            replace(claim, status=status) if claim.claim_role == role else claim
            for claim in package.governed_claims
        ),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-10-01T00:00:00Z"
    )

    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("INVALID_GOVERNED_CLAIM_EVIDENCE",)


@pytest.mark.parametrize("status", _CONTEXT_STATUSES)
@pytest.mark.parametrize(
    "boundary,reason",
    (
        ("source", "INVALID_GOVERNED_CLAIM_EVIDENCE"),
        ("semantic", "UNRESOLVED_SEMANTIC_EVIDENCE"),
        ("rendering", "INVALID_GOVERNED_CLAIM_EVIDENCE"),
        ("qualification", "UNQUALIFIED_HEADLINE_CLAIM"),
    ),
)
def test_context_allowance_preserves_evidence_boundaries(
    status: GovernedClaimStatus, boundary: str, reason: str,
) -> None:
    candidate, package = _context_package(status)
    context = package.governed_claims[-1]
    if boundary == "source":
        context = replace(context, source_ids=("missing-source",))
    elif boundary == "semantic":
        package = replace(
            package,
            resolved_evidence_records=tuple(
                record for record in package.resolved_evidence_records
                if record[0] != context.semantic_relation_evidence_id
            ),
        )
    elif boundary == "rendering":
        context = replace(context, rendered_assertion_zh_hant_hk="London 費用估算仍屬暫定。")
    else:
        package = replace(package, qualification_evidence=())
    package = replace(package, governed_claims=(*package.governed_claims[:-1], context))

    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-10-01T00:00:00Z"
    )

    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == (reason,)


@pytest.mark.parametrize(
    "status",
    (GovernedClaimStatus.ATTRIBUTED_CLAIM_OR_OPINION, GovernedClaimStatus.PUBLISHED_ANALYSIS_OR_FORECAST),
)
def test_other_context_statuses_remain_held(status: GovernedClaimStatus) -> None:
    candidate, package = _context_package(status)

    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-10-01T00:00:00Z"
    )

    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == ("INVALID_GOVERNED_CLAIM_EVIDENCE",)


@pytest.mark.parametrize("status", _CONTEXT_STATUSES)
def test_writer_input_retains_context_status_and_provisional_rendering(
    status: GovernedClaimStatus,
) -> None:
    _, package = _context_package(status)

    context = _writer_evidence_value(package)["approved_governed_claims"][-1]

    assert context["claim_role"] == "CONTEXT"
    assert context["status"] == status.value
    assert context["rendered_assertion"] == "費用估算仍屬暫定。"


@pytest.mark.parametrize("status", _CONTEXT_STATUSES)
@pytest.mark.parametrize("surface", ("v3", "v2", "fixture"))
def test_writer_surfaces_render_and_link_context_exactly_once(
    status: GovernedClaimStatus, surface: str,
) -> None:
    candidate, package = _context_package(status)
    source_renderings = tuple(claim.rendered_assertion_zh_hant_hk for claim in package.governed_claims)
    if surface == "v3":
        # Source order, not the governed inventory order, controls native paragraphs.
        package = replace(package, governed_claims=tuple(reversed(package.governed_claims)))
    if surface == "fixture":
        copy = FixtureWriter().write(candidate, package)
    else:
        title, body, links = required_surface_copy(
            package, paragraphs=True, context_preserving=surface == "v3",
        )
        copy = WriterCopy(
            title=title, body=body, writer_id=f"newsroom.offline-exact-copy.{surface}",
            evidence_package_digest=package.digest, evidence_links=links,
        )

    assert copy.body.count("費用估算仍屬暫定。") == 1
    assert {link.governed_claim_id for link in copy.evidence_links} == {
        claim.claim_id for claim in package.governed_claims
    }
    assert all(result.result == "PASS" for result in validate_writer_copy(copy, package))
    if surface == "v3":
        assert copy.body.split("\n\n") == list(source_renderings)


@pytest.mark.parametrize("status", _CONTEXT_STATUSES)
def test_normal_writer_prompt_includes_context_rendering_and_link(
    status: GovernedClaimStatus,
) -> None:
    candidate, package = _context_package(status)

    prompt = _prompt(candidate, package)

    assert "費用估算仍屬暫定。" in prompt
    assert '"governed_claim_id": "fixture:context"' in prompt


@pytest.mark.parametrize("status", _CONTEXT_STATUSES)
@pytest.mark.parametrize("mutation", ("omitted", "duplicated", "unsupported-residue"))
def test_native_context_copy_keeps_exact_once_and_entailment_fail_closed(
    status: GovernedClaimStatus, mutation: str,
) -> None:
    _, package = _context_package(status)
    title, body, links = required_surface_copy(
        package, paragraphs=True, context_preserving=True,
    )
    if mutation == "omitted":
        body = body.replace("費用估算仍屬暫定。", "")
    elif mutation == "duplicated":
        body += "\n\n費用估算仍屬暫定。"
    else:
        body += "\n\n呢句補充未有證據支持。"
    copy = WriterCopy(
        title=title, body=body, writer_id="newsroom.offline-exact-copy.v3",
        evidence_package_digest=package.digest, evidence_links=links,
    )

    results = {result.validator: result for result in validate_writer_copy(copy, package)}

    assert results["ROLE_SPECIFIC_EXACT_ONCE_STRUCTURE"].result == "FAIL"
    assert results["GOVERNED_CLAIM_ENTAILMENT_BOUNDARY"].result == "FAIL"
    assert results["GOVERNED_CLAIM_ENTAILMENT_BOUNDARY"].reason_code == "UNSUPPORTED_CLAIM_RESIDUE"


@pytest.mark.parametrize(
    "school,rendered",
    (
        ("primary", "該等海報供小學及餐飲承辦商使用，以協助由2027年9月起規劃及提供校園健康食物，惟尚待國會批准。"),
        ("secondary", "這些海報供中學及餐飲承辦商使用，協助由2027年9月起（須待國會批准）規劃及提供校內健康食物。"),
    ),
)
def test_provisional_poster_headline_without_qualification_stays_held(
    school: str, rendered: str,
) -> None:
    # Synthetic normative cases, not claims that the two retained articles qualify.
    candidate, package = _candidate_package()
    headline = (
        f"These posters are for {school} schools and caterers to help with the planning "
        "and provision of healthy food in schools from September 2027, "
        "subject to Parliamentary approval."
    )
    body = (
        "They offer practical guidance on how to apply the school food standards "
        "and make sure healthy options are always available for pupils."
    )
    claims = tuple(
        _bind_fixture_entities(replace(
            claim, claim=text, supporting_excerpt=text, status=status,
            rendered_assertion_zh_hant_hk=rendering,
        ))
        for claim, text, status, rendering in (
            (package.governed_claims[0], headline, GovernedClaimStatus.EXPRESSLY_PROVISIONAL_FACT, rendered),
            (package.governed_claims[1], body, GovernedClaimStatus.CONFIRMED_FACT,
             "該等海報提供實務指引，說明如何應用學校食物標準，並確保學童時刻有健康選擇。"),
        )
    )
    package = replace(
        package, passages=(headline + "\n" + body,), governed_claims=claims,
        substantive_new_information=(headline, body), qualification_evidence=(),
    )

    decision = DeterministicWriteAdmission().decide(
        candidate, package, decided_at="2026-10-01T00:00:00Z"
    )

    assert decision.decision == "HOLD"
    assert decision.stable_reason_codes == (
        "INVALID_GOVERNED_CLAIM_EVIDENCE", "UNQUALIFIED_HEADLINE_CLAIM",
    )
