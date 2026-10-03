"""Generic exact-title recognition; frozen source/receipt replay stays unchanged."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from newsroom.authority.canonical import digest_bytes
from newsroom.control_plane.evidence import (
    NAMED_ENTITY_POLICY_VERSION, NAMED_ENTITY_POLICY_VERSION_V15,
    bounded_named_entities, rendered_named_entities,
)
from newsroom.control_plane.native_assessor import _materialise_reference_result, VERSION
from newsroom.control_plane.native_assessor_spans import (
    PARTITION_VERSION, PARTITION_VERSION_V1, build_lossless_source_view,
)


def _retained():
    return json.loads((Path(__file__).parent / 'fixtures/native_assessor_v22_possessive_title.json').read_text())


@pytest.mark.parametrize('name', [
    "Children's Wellbeing and Schools Act", "Children’s Wellbeing and Schools Act",
    "Teachers' Pension Reform Act", "Teachers’ Pension Reform Act",
])
def test_possessive_is_part_of_the_exact_legal_title(name):
    source = name + ' changed arrangements.'
    names = bounded_named_entities(source)
    assert names == frozenset({(name, 'OFFICIAL_TERM')})
    assert rendered_named_entities('《' + name + '》更改安排。', names) == names
    assert NAMED_ENTITY_POLICY_VERSION.endswith('.v16')
    assert name not in {item[0] for item in bounded_named_entities(source, policy_version=NAMED_ENTITY_POLICY_VERSION_V15)}


def test_empty_quote_residue_is_not_a_new_product_but_unknown_text_stays_visible():
    name = "Children's Wellbeing and Schools Act"
    names = frozenset({(name, 'OFFICIAL_TERM')})
    assert rendered_named_entities('《' + name + '》更改安排。', names) == names
    for text in ('《假冒' + name + '》更改安排。', '《不相關書名》更改安排。'):
        assert rendered_named_entities(text, names) != names
    assert ('《 》', 'PRODUCT') in bounded_named_entities('《 》', policy_version=NAMED_ENTITY_POLICY_VERSION_V15)
    assert bounded_named_entities('《 》') == frozenset()


def test_exact_v22_manifest_and_materialisation_remain_frozen():
    fixture = _retained()
    body = fixture['source_body']
    receipt = fixture['materialisation_receipt']
    assert digest_bytes(body.encode()) == fixture['body_digest'] == receipt['body_digests'][0]
    old = build_lossless_source_view((body,), ('UK-05',), version=PARTITION_VERSION_V1)
    assert old.entity_policy_version == NAMED_ENTITY_POLICY_VERSION_V15
    assert old.manifest_digest == receipt['manifest_digest']
    _material, derived = _materialise_reference_result(fixture['raw_result_text'], old,
        receipt['request_identity'], 'newsroom.native-evidence-assessor.v22')
    assert derived == receipt
    assert 'entity_policy_version' not in old.manifest
    # The actual old result translated Children's: current policy does not
    # rewrite or accept that unsupported rendering merely because names improved.
    source = bounded_named_entities(fixture['claim']['claim'], source_context=body)
    assert rendered_named_entities(fixture['claim']['rendered_assertion_zh_hant_hk'], source) != source


def test_future_view_materialises_the_complete_name_not_a_generated_translation():
    fixture = _retained()
    body = fixture['source_body']
    old = build_lossless_source_view((body,), ('UK-05',), version=PARTITION_VERSION_V1)
    new = build_lossless_source_view((body,), ('UK-05',))
    assert new.partition_version == PARTITION_VERSION
    assert new.entity_policy_version == new.manifest['entity_policy_version'] == NAMED_ENTITY_POLICY_VERSION
    assert old.body_digests == new.body_digests
    assert old.manifest_digest != new.manifest_digest
    assert (("Children's Wellbeing and Schools Act", 'OFFICIAL_TERM'),) in [segment.entities for segment in new.segments]
    wire = deepcopy(json.loads(fixture['raw_result_text']))
    wire['package']['governed_claims'][0]['rendered_assertion_zh_hant_hk_fragments'] = ['第26行所述《', '》更改學校安排。']
    materialised, _receipt = _materialise_reference_result(wire, new, 'future-fixture-only', VERSION)
    claim = materialised['package']['governed_claims'][0]
    names = bounded_named_entities(claim['claim'], source_context=body)
    assert rendered_named_entities(claim['rendered_assertion_zh_hant_hk'], names) == names
    assert "Children's Wellbeing and Schools Act" in claim['rendered_assertion_zh_hant_hk']
