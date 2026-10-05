"""Occurred public-process launches are not enactment of proposed controls."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from newsroom.control_plane.admission import (
    DeterministicWriteAdmission, WriteAdmissionDecision, _decision_id,
    _qualification_relation_is_proven,
)
from newsroom.control_plane.evidence import QualificationEvidence
from newsroom.tests.test_zero_quota_write_loop import _candidate_package


def qualification(span, *, reader_action=None, kind='OFFICIAL_ACTION_OR_DEADLINE'):
    if kind == 'LAW_RIGHT_STATUS_POLICY':
        fields={'change_kind':'PUBLIC_POLICY','change_relation':'NEW_OR_CHANGED_STATE','new_state':span}
    else:
        fields={'action_class':'PROCESS','action_relation':'NEW_OR_CHANGED_OFFICIAL_ACTION','reader_action':reader_action or span}
    return QualificationEvidence(kind,'claim','process-record',tuple({**fields,
        'event_polarity':'AFFIRMED','material_relation_span':span}.items()))


def proven(span, parent, *, source=None, reader_action=None, kind='OFFICIAL_ACTION_OR_DEADLINE'):
    claim=SimpleNamespace(claim=parent,supporting_excerpt=parent)
    return _qualification_relation_is_proven(qualification(span,reader_action=reader_action,kind=kind),
        claim,source_context=parent if source is None else source)


@pytest.mark.parametrize('span,parent,source',[
    ('The government has launched a public consultation',
     'The government has launched a public consultation on proposed licensing controls which could require registration.',None),
    ('A public consultation has been launched',
     'A public consultation has been launched on proposals for new licensing controls.',None),
    ('政府已啟動公眾諮詢','政府已啟動公眾諮詢，就擬議的新規例收集意見。',None),
    ('公眾諮詢已由政府正式展開','公眾諮詢已由政府正式展開，邀請市民提交意見。',None),
    ('The government has launched a consultation',
     'The government has launched a consultation on proposed controls.',
     'The government has launched a consultation on proposed controls. This consultation invites the public to submit views.'),
])
def test_confirmed_public_process_launch_has_source_bound_event(span,parent,source):
    assert proven(span,parent,source=source)
    assert not proven(span,parent,source=source,kind='LAW_RIGHT_STATUS_POLICY')


@pytest.mark.parametrize('source',[
    'Officials deny that the government has launched a public consultation.',
    'If ministers approve; the government has launched a public consultation.',
    'Officials said "the government has launched a public consultation".',
    'The government has launched a public consultation, but this consultation has not started.',
    'The government has launched a public consultation. This statement is false.',
    '假如獲得批准；政府已啟動公眾諮詢。',
    '當局否認政府已啟動公眾諮詢。',
    '「政府已啟動公眾諮詢」這項說法不實。',
])
def test_clipped_witness_cannot_excise_parent_negation_modality_or_quote(source):
    span='政府已啟動公眾諮詢'if '政府' in source else 'the government has launched a public consultation'
    assert not proven(span,span,source=source)


@pytest.mark.parametrize('parent',[
    'The government plans to launch a public consultation.',
    'A public consultation will be launched next year.',
    'The government could have launched a public consultation.',
    '政府計劃啟動公眾諮詢。',
    '公眾諮詢將由政府展開。',
    'The government has launched an internal consultation with cabinet staff.',
])
def test_future_or_internal_process_is_not_confirmed_public_reader_action(parent):
    assert not proven(parent,parent)


def test_reader_action_and_public_antecedent_must_bind_exact_current_source():
    parent='The government has launched a consultation on proposed controls.'
    span='The government has launched a consultation'
    assert not proven(span,parent)
    assert not proven(span,parent,source=parent+' Residents submit views on a different transport consultation.')
    assert not proven(span,parent,reader_action='submit licensing applications')
    assert not proven(span,parent,source='The authority is considering a different process.')


@pytest.mark.parametrize('invitation', [
    'It seeks views from the public, industry and business to gauge how these changes would affect them.',
    'This consultation seeks views from the public, industry and business.',
])
def test_separate_reader_action_binds_the_confirmed_process_and_public_invitation(invitation):
    launch='The changes will be subject to a consultation which was launched yesterday.'
    parent=launch+' '+invitation
    assert proven(launch,parent,reader_action='seeks views from the public, industry and business')


@pytest.mark.parametrize('suffix', [
    'It will seek views from the public next year.',
    'It does not seek views from the public.',
    'If funding is granted, it seeks views from the public.',
    'It seeks views from the public on a different consultation.',
    'A separate programme was launched. It seeks views from the public.',
])
def test_reader_action_cannot_excise_invitation_modality_or_change_antecedent(suffix):
    launch='The changes will be subject to a consultation which was launched yesterday.'
    parent=launch+' '+suffix
    assert not proven(launch,parent,reader_action='seeks views from the public')


def test_reader_action_phrase_must_be_in_the_same_claim_and_source():
    launch='The changes will be subject to a consultation which was launched yesterday.'
    invitation='It seeks views from the public, industry and business.'
    assert not proven(launch,launch,source=launch+' '+invitation,
        reader_action='seeks views from the public, industry and business')


@pytest.mark.parametrize('parent', [
    'The government has launched a consultation alongside a survey.',
    'The government has launched a consultation and another consultation.',
])
def test_pronoun_invitation_requires_one_unambiguous_process_in_parent(parent):
    span='The government has launched a consultation'
    text=parent+' It seeks views from the public.'
    assert not proven(span,text,reader_action='seeks views from the public')


def test_process_truth_does_not_supply_missing_substantive_newness():
    parent='The government has launched a public consultation on proposed controls.'
    assert proven('The government has launched a public consultation',parent)
    candidate,package=_candidate_package()
    decision=DeterministicWriteAdmission().decide(candidate,replace(package,substantive_new_information=()),
        decided_at='2026-10-05T00:00:00Z')
    assert decision.decision=='REJECT' and decision.stable_reason_codes==('NO_SUBSTANTIVE_NEW_INFORMATION',)


@pytest.mark.parametrize('previous_relation', ['v3', 'v4'])
def test_prior_v10_relation_decision_reads_without_becoming_current(tmp_path, previous_relation):
    from newsroom.control_plane import admission
    candidate,package=_candidate_package()
    current=DeterministicWriteAdmission().decide(candidate,package,decided_at='2026-10-05T00:00:00Z')
    record=current.as_record()
    record['policy_version']=current.policy_version.rsplit('+',1)[0]+'+newsroom.qualification-relation.'+previous_relation
    values={key:getattr(current,key)for key in record if key not in {'decision_id','decided_at'}}
    values['policy_version']=record['policy_version'];record['decision_id']=_decision_id(**values)
    assert WriteAdmissionDecision.from_record(record).as_record()==record
    assert current.policy_version.endswith('newsroom.qualification-relation.v5')
    assert record['policy_version']!=admission.WRITE_ADMISSION_POLICY_VERSION
    from newsroom.control_plane.store import connect, retain_write_admission_decision
    old=WriteAdmissionDecision.from_record(record)
    path=str(tmp_path/'retained-admission.sqlite3')
    connection=connect(path)
    retain_write_admission_decision(connection,old)
    retain_write_admission_decision(connection,current)
    connection.commit()
    connection.close()
    connection=connect(path)
    retain_write_admission_decision(connection,old)
    rows=connection.execute('SELECT record_json FROM unpublished_write_admission_decisions WHERE decision_id=?',(old.decision_id,)).fetchall()
    import json
    assert len(rows)==1 and WriteAdmissionDecision.from_record(json.loads(rows[0][0]))==old
    assert connection.execute('SELECT count(*) FROM unpublished_write_admission_decisions').fetchone()[0]==2
    connection.close()
    with pytest.raises(ValueError,match='unsupported write-admission'):
        WriteAdmissionDecision.from_record({**record,'policy_version':record['policy_version'].replace('relation.'+previous_relation,'relation.v99')})


def test_context_denial_cannot_fall_back_to_generic_process_keywords():
    span='The government has launched a public consultation process'
    assert not proven(span,span,source='Officials deny that '+span+'.')


@pytest.mark.parametrize('span,parent',[
    ('a public consultation which was launched yesterday',
     'The changes will be subject to a public consultation which was launched yesterday.'),
    ('The changes will be subject to a public consultation which was launched yesterday',
     'The changes will be subject to a public consultation which was launched yesterday.'),
])
def test_future_policy_target_does_not_negate_closed_relative_process_event(span,parent):
    assert proven(span,parent)
    assert not proven(span,parent,kind='LAW_RIGHT_STATUS_POLICY')


@pytest.mark.parametrize('parent',[
    'The changes will be subject to a public consultation which will be launched next year.',
    'The changes will be subject to a public consultation if it is launched.',
    'If approval is granted, a public consultation was launched yesterday.',
    'Officials deny that a public consultation was launched yesterday.',
    'The changes will be subject to a public consultation which was not launched yesterday.',
])
def test_relative_process_future_condition_or_denial_is_not_exempt(parent):
    assert not proven(parent,parent)


def test_current_package_admits_confirmed_process_but_not_clipped_denial():
    from newsroom.tests.test_zero_quota_write_loop import _bind_fixture_entities
    parent='The changes will be subject to a public consultation which was launched yesterday.'
    candidate,package=_candidate_package()
    claim=_bind_fixture_entities(replace(package.governed_claims[1],claim=parent,supporting_excerpt=parent,
        rendered_assertion_zh_hant_hk='政府已展開公眾諮詢，市民可提交意見；新措施尚未生效。'))
    q=qualification(parent)
    q=replace(q,governed_claim_id=claim.claim_id,qualification_record_id=package.qualification_evidence[1].qualification_record_id)
    package=replace(package,passages=(package.passages[0]+'\n'+parent,),
        governed_claims=(package.governed_claims[0],claim),
        substantive_new_information=(package.governed_claims[0].claim,parent),
        qualification_evidence=(package.qualification_evidence[0],q),
        resolved_evidence_records=(*package.resolved_evidence_records,
            *((record,'fixture-entity-digest')for _,_,record in claim.named_entity_evidence)))
    accepted=DeterministicWriteAdmission().decide(candidate,package,decided_at='2026-10-05T00:00:00Z')
    assert accepted.decision=='WRITE_READY',accepted.stable_reason_codes
    denied=replace(package,passages=(package.passages[0].replace(parent,'Officials deny that '+parent),))
    result=DeterministicWriteAdmission().decide(candidate,denied,decided_at='2026-10-05T00:00:00Z')
    assert result.decision=='HOLD' and result.stable_reason_codes==('QUALIFICATION_EVIDENCE_NOT_EXACT',)


def test_current_package_admits_exact_separate_public_reader_action():
    from newsroom.tests.test_zero_quota_write_loop import _bind_fixture_entities
    launch='The changes will be subject to a consultation which was launched yesterday.'
    text=launch+' It seeks views from the public, industry and business.'
    candidate,package=_candidate_package()
    claim=_bind_fixture_entities(replace(package.governed_claims[1],claim=text,supporting_excerpt=text,
        rendered_assertion_zh_hant_hk='諮詢已於昨日展開，向公眾、業界及企業徵求意見。'))
    q=replace(qualification(launch,reader_action='seeks views from the public, industry and business'),
        governed_claim_id=claim.claim_id,qualification_record_id=package.qualification_evidence[1].qualification_record_id)
    package=replace(package,passages=(package.passages[0]+'\n'+text,),
        governed_claims=(package.governed_claims[0],claim),
        substantive_new_information=(package.governed_claims[0].claim,text),
        qualification_evidence=(package.qualification_evidence[0],q),
        resolved_evidence_records=(*package.resolved_evidence_records,
            *((record,'fixture-entity-digest')for _,_,record in claim.named_entity_evidence)))
    result=DeterministicWriteAdmission().decide(candidate,package,decided_at='2026-10-05T00:00:00Z')
    assert result.decision=='WRITE_READY',result.stable_reason_codes
