"""Partition choice is retained per request, never inferred for old receipts."""

import pytest

from newsroom.control_plane import native_assessor as native
from newsroom.control_plane.native_assessor_references import build_source_view
from newsroom.control_plane.native_assessor_spans import PARTITION_VERSION, build_lossless_source_view
from newsroom.control_plane.native_evidence import NativeEvidenceError

BODY = 'The official deadline changed. The government announced the new deadline.'


def test_partition_binding_leaves_historical_reference_bytes_unchanged():
    old = build_source_view((BODY,), ('UK-01',))
    assert native._reference_binding(old) == {
        'version': native.SOURCE_REFERENCE_VERSION,
        'manifest_digest': old.manifest_digest,
        'body_digests': list(old.body_digests),
    }
    new = build_lossless_source_view((BODY,), ('UK-01',))
    assert native._reference_binding(new)['partition_version'] == PARTITION_VERSION
    assert new.manifest_digest != old.manifest_digest
    assert new.body_digests == old.body_digests


@pytest.mark.parametrize('partitioned', (False, True))
def test_retained_partition_reconstructs_only_its_exact_request_view(partitioned):
    builder = build_lossless_source_view if partitioned else build_source_view
    expected = builder((BODY,), ('UK-01',))
    binding = native._reference_binding(expected)
    rebuilt = native._source_view_for_binding((BODY,), ('UK-01',), binding)
    assert rebuilt == expected
    assert rebuilt.passages == (BODY,)
    assert len(rebuilt.segments) == (2 if partitioned else 1)


@pytest.mark.parametrize('change', (
    {'partition_version': 'unknown-partition'},
    {'partition_version': []},
    {'manifest_digest': 'sha256:' + '0' * 64},
    {'body_digests': ['sha256:' + '0' * 64]},
))
def test_unknown_partition_or_changed_source_binding_is_rejected(change):
    view = build_source_view((BODY,), ('UK-01',))
    binding = native._reference_binding(view) | change
    with pytest.raises(NativeEvidenceError):
        native._source_view_for_binding((BODY,), ('UK-01',), binding)
