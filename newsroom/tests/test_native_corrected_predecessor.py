"""Current acknowledged corrections, not original receipt slots, are predecessors."""
from types import SimpleNamespace as NS
import pytest
from newsroom.control_plane.native_publication import NativePublicationContinuation, NativePublicationError

REFS=('story_event_id','publication_event_id','delivery_attempt_event_id','delivery_evidence_event_id')

def test_exact_corrected_ack_is_selected_without_rewriting_original_facts():
    old=dict(zip(REFS,('old-story','old-pub','old-attempt','old-evidence')))
    corrected=dict(zip(REFS,('new-story','new-pub','new-attempt','new-evidence')))
    facts={**old,'candidate_id':'candidate','factual_correction_result':corrected}
    reads=[]
    def read(refs,*,proof):
        reads.append(dict(refs));new=refs['story_event_id']=='new-story'
        return NS(story_receipt=NS(aggregate_version=6 if new else 5),
                  attempt_receipt=NS(publication_id='publication',aggregate_version=12 if new else 10)),NS(story_id='story')
    continuation=object.__new__(NativePublicationContinuation)
    continuation._runtime=NS(publication=NS(read_acknowledged=read),proof=object())
    before={**facts,'factual_correction_result':dict(corrected)}
    selected=continuation._acknowledged_current_facts(facts)
    assert selected=={**facts,**corrected}
    assert facts==before
    assert reads==[old,corrected]
    continuation._journal=NS(iter_summaries=lambda:iter((('prior',{'stage':'ACKNOWLEDGED','facts':facts}),)))
    continuation._runtime.policies=NS(publication=NS(editorial_story_command_definition_digest='story-contract',serving_attempt_command_definition_digest='attempt-contract'))
    events=[]
    def prior_event(identity,**_):
        events.append(identity)
        return NS(aggregate_version=6 if identity=='new-story' else 12)
    continuation._prior_event=prior_event
    assert continuation._prior_acknowledged_versions(revision_id='incoming',candidate_id='candidate')==(6,12)
    assert events==['new-story','new-attempt']
    assert facts==before

@pytest.mark.parametrize('fault',('missing','wrong_story','wrong_publication','older','older_attempt'))
def test_incomplete_or_unrelated_correction_is_not_a_version_shortcut(fault):
    old=dict(zip(REFS,('old-story','old-pub','old-attempt','old-evidence')))
    corrected=dict(zip(REFS,('new-story','new-pub','new-attempt','new-evidence')))
    if fault=='missing':corrected.pop('delivery_evidence_event_id')
    def read(refs,*,proof):
        new=refs['story_event_id']=='new-story'
        return NS(story_receipt=NS(aggregate_version=4 if new and fault=='older' else 6 if new else 5),
                  attempt_receipt=NS(publication_id='wrong'if new and fault=='wrong_publication'else'publication',
                    aggregate_version=9 if new and fault=='older_attempt'else 12 if new else 10)),NS(story_id='wrong'if new and fault=='wrong_story'else'story')
    continuation=object.__new__(NativePublicationContinuation)
    continuation._runtime=NS(publication=NS(read_acknowledged=read),proof=object())
    with pytest.raises(NativePublicationError):continuation._acknowledged_current_facts({**old,'factual_correction_result':corrected})
