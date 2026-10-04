"""Lossless wire lower bounds reject known overflow before contextual NER."""
from contextlib import nullcontext
from dataclasses import replace
import sqlite3

import pytest

from newsroom.control_plane import native_assessor as assessor
from newsroom.control_plane.native_evidence import NativeEvidenceHold
from newsroom.tests.test_native_assessor import _usage
from newsroom.tests.assessor_fixture_support import candidate_fixture
from newsroom.tests.test_increment10_editorial import _ready_package
from newsroom.increment10.evidence import _base_package


def test_csv_known_wire_overflow_never_builds_entities_or_allocates(tmp_path,monkeypatch):
    connection,_,candidate=candidate_fixture(tmp_path)
    service,usage=_usage(tmp_path,monkeypatch)
    body=''.join(f'Row {n}: A="'+('x'*165)+'"\n'for n in range(1,231))
    base=replace(_base_package(_ready_package(candidate)[1]),passages=(body,),source_ids=('UK-01',))
    # Raw source bytes fit; mandatory per-row wire framing alone does not.
    bound=assessor.native_assessment_input_bound(usage._policy)['max_request_bytes']
    from newsroom.control_plane.native_assessor_spans import source_wire_lower_bound_bytes
    assert len(body.encode())<bound<source_wire_lower_bound_bytes(base.passages,base.source_ids)
    monkeypatch.setattr(assessor,'build_lossless_source_view',lambda *a,**k:pytest.fail('over-bound source built contextual entities'))
    subject=assessor.AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('provider dispatched'),usage=usage,dispatch_fence=nullcontext)
    try:
        with pytest.raises(NativeEvidenceHold) as held:
            subject(candidate,base,(),())
        assert held.value.reason_code=='ASSESSOR_EXACT_INPUT_BOUND_HOLD'
        with sqlite3.connect(service.path) as c:
            assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==0
            assert c.execute('SELECT count(*) FROM model_work_envelopes').fetchone()[0]==0
    finally:connection.close()


@pytest.mark.parametrize('body',(
    'Row 1: A=plain\n',
    'Row 1: A="Hong Kong 中文 😀"; B=\\value\r\nRow 2: A="quote\\\""\n',
    'Row 1: A="tabs\tcontrol\x01"\u2028Row 2: A=é\r',
))
@pytest.mark.parametrize('version',('newsroom.native-assessor-spans.v1','newsroom.native-assessor-spans.v2'))
def test_csv_lower_bound_matches_empty_entity_wire_utf8_and_partition(body,version):
    from newsroom.authority.canonical import canonical_json_bytes
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view,source_wire_lower_bound_bytes
    view=build_lossless_source_view((body,),('UK-01',),version=version)
    # Contextual names may add entity bytes; the complete actual wire is the
    # independent decoder/encoder oracle, never a guessed character count.
    records=[{**s.request_record(),'rendering_fragment_count':len(s.entities)+1}for s in view.segments]
    actual=len(canonical_json_bytes(records))
    lower=source_wire_lower_bound_bytes((body,),('UK-01',))
    assert lower<=actual
    if not any(s.entities for s in view.segments):assert lower==actual
    assert ''.join(s.text for s in view.segments)==body


@pytest.mark.parametrize('body',(
    'Sentence one. Sentence two.\nRow 1: A="a.b!"\n',
    '中文句一。下一句！\nRow 5: B=😀\n',
    'Row 01: not canonical CSV. More text.\n',
))
def test_mixed_partition_lower_bound_never_exceeds_exact_wire(body):
    from newsroom.authority.canonical import canonical_json_bytes
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view,source_wire_lower_bound_bytes
    view=build_lossless_source_view((body,),('source-"😀',))
    records=[{**s.request_record(),'rendering_fragment_count':len(s.entities)+1}for s in view.segments]
    assert source_wire_lower_bound_bytes((body,),('source-"😀',))<=len(canonical_json_bytes(records))


@pytest.mark.parametrize('difference',(-1,0,1))
def test_lower_bound_strict_boundary_preserves_below_and_equal_original_view(tmp_path,monkeypatch,difference):
    from newsroom.control_plane.native_assessor_spans import source_wire_lower_bound_bytes
    connection,_,candidate=candidate_fixture(tmp_path);service,usage=_usage(tmp_path,monkeypatch)
    body='Row 1: A=plain\n';base=replace(_base_package(_ready_package(candidate)[1]),passages=(body,),source_ids=('UK-01',))
    lower=source_wire_lower_bound_bytes(base.passages,base.source_ids)
    monkeypatch.setattr(assessor,'native_assessment_input_bound',lambda _: {'max_request_bytes':lower+difference})
    class OriginalViewReached(Exception):pass
    def original_view(*args,**kwargs):
        assert args==(base.passages,base.source_ids)
        raise OriginalViewReached
    monkeypatch.setattr(assessor,'build_lossless_source_view',original_view)
    subject=assessor.AutonomousNativeEvidenceAssessor(lambda _:pytest.fail('provider dispatched'),usage=usage,dispatch_fence=nullcontext)
    try:
        if difference<0:
            with pytest.raises(NativeEvidenceHold,match='ASSESSOR_EXACT_INPUT_BOUND_HOLD'):
                subject(candidate,base,(),())
        else:
            with pytest.raises(OriginalViewReached):subject(candidate,base,(),())
        with sqlite3.connect(service.path) as c:
            assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==0
    finally:connection.close()


@pytest.mark.parametrize('second',('Row 2: B="中文 😀"\n','Non-CSV sentence. Another sentence.\n'))
def test_two_source_lower_bound_uses_each_request_source_array(second):
    from newsroom.authority.canonical import canonical_json_bytes
    from newsroom.control_plane.native_assessor_spans import build_lossless_source_view,source_wire_lower_bound_bytes
    passages=('Prefix sentence. Split sentence!\nRow 1: A="\\quoted\\""\n',second)
    source_ids=('source-"一','source-\\😀')
    view=build_lossless_source_view(passages,source_ids)
    # This is the actual request's per-source arrays, not a flattened view array.
    sources=[{'source_id':identity,'segments':[
        {**segment.request_record(),'rendering_fragment_count':len(segment.entities)+1}
        for segment in view.segments if segment.source_id==identity]}
        for identity in source_ids]
    exact_arrays=sum(len(canonical_json_bytes(source['segments']))for source in sources)
    lower=source_wire_lower_bound_bytes(passages,source_ids)
    assert lower<=exact_arrays<=len(canonical_json_bytes({'sources':sources}))
    assert any(segment.span_id.startswith('S2L')for segment in view.segments)
    assert ''.join(segment.text for segment in view.segments if segment.source_id==source_ids[1])==second
