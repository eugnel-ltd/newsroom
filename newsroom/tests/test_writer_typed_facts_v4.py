"""Writer dispatches the shared typed fact check only for v2 Source rendering."""
from types import SimpleNamespace
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
