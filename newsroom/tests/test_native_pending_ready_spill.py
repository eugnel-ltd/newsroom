"""Newly complete pending work reaches the existing ready-spill quantum."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from newsroom.control_plane.native_graphiti import NativeGraphitiOutcome
from newsroom.tests.test_native_pipeline import _open,_overrunning_ready_spill
from newsroom.tests.test_native_graphiti import _native


def test_new_graphiti_completion_uses_existing_spill_after_fresh_quantum(tmp_path,monkeypatch):
    pipeline,journal,connection,units,calls,dispositions=_open(tmp_path,monkeypatch)
    now=[0.0];pipeline._monotonic_clock=lambda:now[0]
    current=replace(units[0],updated_at='2026-10-04T06:30:00Z')
    dispositions[0]=(NS(source_id=current.source_id,status='READY',reason_code='RETAINED',units=(current,)),)
    phases=[]
    original=pipeline._advance_revisions
    def advance(revisions,*,work_deadline):
        phases.append((tuple(r for r,_ in revisions),work_deadline))
        return original(revisions,work_deadline=work_deadline)
    def graphiti(selected,**kwargs):
        assert len(selected)==1
        now[0]+=301
        return (NativeGraphitiOutcome(current.ingest_id,'GRAPHITI_COMPLETE',current.digest,None),)
    pipeline._advance_revisions=advance;pipeline._graphiti=NS(advance=graphiti)
    try:
        pipeline.tick(cycle_id='completed-at-fresh-deadline')
        assert journal.current(current.revision_id)['stage']=='ACKNOWLEDGED'
        assert [r for kind,r in calls if kind=='publish']==[current.revision_id]
        assert len(phases)==3 # Existing ordinary/fresh/reassessment turns only.
        assert phases[-1]==((current.revision_id,),601)
        assert pipeline._spill_archive_turn is True
    finally:connection.close()


@pytest.mark.parametrize('archive_turn',[False,True])
def test_pending_ready_spill_shares_current_archive_order_and_atomic_budget(tmp_path,monkeypatch,archive_turn):
    pipeline,journal,connection,calls,now=_overrunning_ready_spill(tmp_path,monkeypatch)
    pipeline._spill_archive_turn=archive_turn
    archives=[]
    for number,count in enumerate((25,94)):
        first=replace(_native(f'archive-{number}'),updated_at=f'202{number}-01-01T00:00:00Z',chunk_count=count)
        chunks=tuple(replace(first,chunk_ordinal=ordinal) for ordinal in range(1,count+1))
        journal.land(chunks);journal.advance(first.revision_id,stage='GRAPHITI_COMPLETE',facts={'graphiti_receipts':[{'retained':True}]})
        archives.append(first)
    current=replace(_native('current-WTS'),updated_at='2026-10-04T06:30:00Z')
    pipeline._intake=NS(poll=lambda:(NS(source_id=current.source_id,status='READY',reason_code='RETAINED',units=(current,)),))
    def graphiti(selected,**kwargs):
        now[0]+=301
        return tuple(NativeGraphitiOutcome(unit.ingest_id,'GRAPHITI_COMPLETE',unit.digest,None) for unit in selected)
    pipeline._graphiti=NS(advance=graphiti)
    try:
        pipeline.tick(cycle_id='same-spill-preference')
        publications=[r for kind,r in calls if kind=='publish']
        assert publications==[archives[0].revision_id if archive_turn else current.revision_id]
        assert len(publications)==len(set(publications))
        assert pipeline._spill_archive_turn is not archive_turn
        unselected=current if archive_turn else archives[0]
        assert journal.current(unselected.revision_id)['stage']=='GRAPHITI_COMPLETE'
    finally:connection.close()
