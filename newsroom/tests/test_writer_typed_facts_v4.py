"""Writer dispatches the shared typed fact check only for v2 Source rendering."""
from types import SimpleNamespace
import pytest
from newsroom.control_plane.evidence import SOURCE_RENDERING_CONTRACT, SOURCE_RENDERING_CONTRACT_V2
from newsroom.control_plane.writer import _typed_claim_numeric_relation


def claim(source, rendered, *, contract=SOURCE_RENDERING_CONTRACT_V2):
    return SimpleNamespace(claim=source, rendered_assertion_zh_hant_hk=rendered,
        localised_factual_expressions=(), named_entities=(),
        source_rendering_ref=(('contract', contract),))


def test_writer_typed_contract_does_not_invent_a_same_customer_quantity():
    assert _typed_claim_numeric_relation(claim('Use the same customer help email address.', '沿用同一客戶支援電郵地址。')) is True
    assert _typed_claim_numeric_relation(claim('Use the same customer help email address.', '沿用同一客戶支援電郵地址。',
                                              contract=SOURCE_RENDERING_CONTRACT)) is None


def test_writer_typed_contract_rejects_changed_embedded_quantity():
    assert _typed_claim_numeric_relation(claim('計劃為期2年。', '計劃為期3年。')) is False


def test_writer_typed_contract_does_not_call_modal_may_a_calendar_month():
    assert _typed_claim_numeric_relation(claim('The scheme may close.', '計劃可能停止。')) is True


@pytest.mark.parametrize('rendered,expected', [('關閉時間為120分鐘。','PASS'), ('關閉時間為180分鐘。','FAIL')])
def test_writer_accepts_typed_unit_conversion_without_requiring_a_redundant_pair(rendered, expected):
    from dataclasses import replace
    from newsroom.authority.canonical import digest_bytes
    from newsroom.tests.test_document_year_fidelity import package, checks
    value = package()
    original = value.governed_claims[0]
    ref = tuple(sorted({'contract': SOURCE_RENDERING_CONTRACT_V2, 'operation': 'SOURCE_RENDERING',
        'invocation_id': digest_bytes(b'fixture'),
        'raw_admission_id': '00000000-0000-4000-8000-000000000001',
        'receipt_admission_id': '00000000-0000-4000-8000-000000000002'}.items()))
    converted = replace(original, claim='The closure lasts 2 hours.', supporting_excerpt='The closure lasts 2 hours.',
        rendered_assertion_zh_hant_hk=rendered, named_entities=(),
        named_entity_evidence=(), rendered_named_entities=(), localised_factual_expressions=(),
        source_rendering_ref=ref)
    value = replace(value, governed_claims=(converted, *value.governed_claims[1:]),
        substantive_new_information=tuple(converted.claim if text == original.claim else text
                                           for text in value.substantive_new_information),
        passages=tuple(text.replace(original.claim, converted.claim) for text in value.passages))
    # Only the numeric component is claimed here, not forged-reference admission.
    assert checks(value)['NUMERIC_AND_DATE_FIDELITY'] == expected
