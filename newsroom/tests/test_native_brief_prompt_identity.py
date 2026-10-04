"""Prompt clarity applies to new intent without respending retained packages."""
import json
import sqlite3

import pytest

from newsroom.authority.canonical import digest_bytes,canonical_json_bytes,digest_canonical
from newsroom.control_plane import native_story_model as model_module
from newsroom.control_plane import native_story_writer as writer
from newsroom.control_plane.model_usage import ModelUsageAdmissionError
from newsroom.tests.test_native_story_model import _model,SCHEMA

IDENTITIES={'candidate_id':'candidate-1','hypothesis_digest':'sha256:'+'c'*64,'admission_decision_id':'decision-1'}
REQUEST={'source_package_digest':'sha256:'+'b'*64,'facts':'approved facts'}


def test_future_draft_uses_new_prompt_and_same_package_replays_legacy_zero_calls(tmp_path,monkeypatch):
    model,service,calls=_model(tmp_path,monkeypatch)
    original=writer.LEGACY_DRAFT_SYSTEM
    model.call(REQUEST,phase='DRAFT',schema=SCHEMA,system=original,**IDENTITIES)
    with service._connection() as c:old_allocation=c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]
    selected=model._draft_system(REQUEST['source_package_digest'],**IDENTITIES)
    assert selected==writer.LEGACY_DRAFT_SYSTEM==original
    assert model.call(REQUEST,phase='DRAFT',schema=SCHEMA,system=selected,**IDENTITIES)=={'text':'supported copy'}
    assert len(calls)==1
    with service._connection() as c:assert c.execute('SELECT count(*) FROM model_invocation_allocations').fetchone()[0]==old_allocation
    assert model._draft_system('sha256:'+'e'*64,**IDENTITIES)==writer.DRAFT_SYSTEM
    assert writer.DRAFT_SYSTEM!=writer.LEGACY_DRAFT_SYSTEM


def test_unknown_retained_system_digest_holds_without_new_call(tmp_path,monkeypatch):
    model,service,calls=_model(tmp_path,monkeypatch)
    model.call(REQUEST,phase='DRAFT',schema=SCHEMA,system='unrecognised instruction',**IDENTITIES)
    with pytest.raises(ModelUsageAdmissionError,match='prompt'):
        model._draft_system(REQUEST['source_package_digest'],**IDENTITIES)
    assert len(calls)==1


def test_future_prompt_is_hashed_and_new_package_does_not_upgrade_old_key(tmp_path,monkeypatch):
    model,service,calls=_model(tmp_path,monkeypatch)
    selected=model._draft_system(REQUEST['source_package_digest'],**IDENTITIES)
    assert selected==writer.DRAFT_SYSTEM
    model.call(REQUEST,phase='DRAFT',schema=SCHEMA,system=selected,**IDENTITIES)
    with service._connection() as c:
        context=json.loads(c.execute('SELECT record_json FROM model_invocation_context_manifests').fetchone()[0])
        key,stored_input=c.execute('SELECT request_key,input_digest FROM native_story_model_results').fetchone()
    assert context['system_digest']==digest_bytes(writer.DRAFT_SYSTEM.encode())
    assert stored_input==digest_canonical({'request':REQUEST,'system':writer.DRAFT_SYSTEM,'schema':SCHEMA})
    assert key==digest_canonical({'version':model_module.VERSION,'package':REQUEST['source_package_digest'],'phase':'DRAFT'})
    assert writer.WRITER_ID=='newsroom.native-story-writer.v1'
    assert model_module.VERSION=='newsroom.native-story-model.v1'
    assert len(calls)==1


def test_allocated_unsettled_legacy_intent_is_never_redispatched(tmp_path,monkeypatch):
    model,service,calls=_model(tmp_path,monkeypatch)
    model.call(REQUEST,phase='DRAFT',schema=SCHEMA,system=writer.LEGACY_DRAFT_SYSTEM,**IDENTITIES)
    # Result absence models a crash after accounted intent; its allocation and
    # terminal remain real and immutable, rather than fabricated retry evidence.
    with service._connection() as c:c.execute('DELETE FROM native_story_model_results')
    system=model._draft_system(REQUEST['source_package_digest'],**IDENTITIES)
    assert system==writer.LEGACY_DRAFT_SYSTEM
    with pytest.raises(ModelUsageAdmissionError):
        model.call(REQUEST,phase='DRAFT',schema=SCHEMA,system=system,**IDENTITIES)
    assert len(calls)==1


@pytest.mark.parametrize('damage',('missing_context','corrupt_context','changed_facts'))
def test_invalid_retained_prompt_or_changed_request_remains_no_retry(tmp_path,monkeypatch,damage):
    model,service,calls=_model(tmp_path,monkeypatch)
    model.call(REQUEST,phase='DRAFT',schema=SCHEMA,system=writer.LEGACY_DRAFT_SYSTEM,**IDENTITIES)
    if damage=='changed_facts':
        system=model._draft_system(REQUEST['source_package_digest'],**IDENTITIES)
        with pytest.raises(ModelUsageAdmissionError):
            model.call({**REQUEST,'facts':'changed'},phase='DRAFT',schema=SCHEMA,system=system,**IDENTITIES)
    else:
        with service._connection() as c:
            if damage=='missing_context':
                c.execute('PRAGMA foreign_keys=OFF')
                c.execute('DELETE FROM model_invocation_context_manifests')
            else:c.execute("UPDATE model_invocation_context_manifests SET record_json='{}'")
        with pytest.raises(ModelUsageAdmissionError):
            model._draft_system(REQUEST['source_package_digest'],**IDENTITIES)
    assert len(calls)==1


def test_write_draft_callback_uses_selected_prompt_and_review_is_unchanged(tmp_path,monkeypatch):
    from copy import deepcopy
    from newsroom.tests.test_native_story_writer import _package,DRAFT,_review
    model,_,_=_model(tmp_path,monkeypatch);seen=[]
    monkeypatch.setattr(model,'_draft_system',lambda digest,**ids:writer.LEGACY_DRAFT_SYSTEM)
    def retained_call(request,*,phase,schema,system,**identities):
        seen.append((phase,system))
        return deepcopy(DRAFT)if phase=='DRAFT'else _review(request)
    monkeypatch.setattr(model,'call',retained_call)
    result=model.write(_package(),**IDENTITIES)
    assert all(check.result=='PASS'for check in result.validators)
    assert seen==[('DRAFT',writer.LEGACY_DRAFT_SYSTEM),('REVIEW',writer.REVIEW_SYSTEM)]
