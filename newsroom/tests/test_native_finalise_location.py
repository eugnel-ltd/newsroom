"""Optional phase diagnostics identify failures without retaining their content."""
import pytest
from newsroom.control_plane import native_graphiti


def test_phase_failure_reports_only_innermost_code_location(monkeypatch):
    records=[]
    monkeypatch.setattr(native_graphiti,'emit_diagnostic',lambda event,data:records.append((event,data)))
    failure=RuntimeError('PRIVATE_SOURCE_AND_PROVIDER_TEXT')
    def exact_failure():raise failure
    with pytest.raises(RuntimeError) as caught:
        with native_graphiti._native_phase('FINALISE',cycle_id='location',cohort_count=1):exact_failure()
    assert caught.value is failure
    _,record=records[0]
    assert record['file']=='test_native_finalise_location.py'
    assert record['function']=='exact_failure'
    assert isinstance(record['line'],int)
    assert 'PRIVATE_SOURCE_AND_PROVIDER_TEXT' not in str(record)


def test_success_has_no_failure_location(monkeypatch):
    records=[]
    monkeypatch.setattr(native_graphiti,'emit_diagnostic',lambda _,data:records.append(data))
    with native_graphiti._native_phase('FINALISE',cycle_id='success',cohort_count=0):pass
    assert records[0]['status']=='COMPLETE'
    assert not any(key in records[0]for key in ('file','function','line'))
