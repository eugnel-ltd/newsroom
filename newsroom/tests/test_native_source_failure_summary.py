"""Source failure observations group safely without changing intake outcomes."""
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import UTC, datetime
import json
from types import SimpleNamespace as NS
import urllib.error

import pytest

from newsroom.authority import AuthorityPersistenceError
from newsroom.control_plane import native_source_intake as n
from newsroom.control_plane.items import SourceItem
from newsroom.control_plane.veto import VetoError

NOW = datetime(2026, 10, 4, tzinfo=UTC)


def _failed(error):
    future = Future()
    try:
        raise error
    except Exception as caught:
        future.set_exception(caught)
    return future


def _intake(monkeypatch, source='UK-03'):
    intake = object.__new__(n.NativeSourceIntake)
    intake._pending_units = {}
    intake._fence = lambda *_: nullcontext()
    intake._fetch = lambda _: (200, b'{}')
    intake._clock = lambda: NOW
    intake._admit_observation = lambda *a, **k: ('admission', NS(access_decision_id='access'))
    monkeypatch.setattr(n, 'SOURCE_IDS', (source,))
    events = []
    monkeypatch.setattr(n, 'emit_diagnostic', lambda event, data: events.append((event, data)), raising=False)
    return intake, events


def _manual(intake, monkeypatch, errors):
    sections = tuple((f'/guidance/section-{i}', f'Section {i}') for i in range(len(errors)))
    monkeypatch.setattr(n, 'parse_govuk_manual_inventory', lambda *a, **k: NS(sections=sections))
    intake._fetch_manual_sections = lambda *_: ((SourceItem('UK-03', str(i), title, title,
        'https://www.gov.uk'+path), _failed(error)) for i, ((path, title), error) in enumerate(zip(sections, errors)))
    intake._poll_one = lambda source: intake._poll_direct_govuk(source, 'definition', 'version', NS(), NS(record_id='rights'))


def test_one_hundred_identical_manual_failures_emit_one_group_and_reset_next_poll(monkeypatch):
    intake, events = _intake(monkeypatch)
    _manual(intake, monkeypatch, [ValueError('https://secret.invalid/?TOKEN=hidden source-body') for _ in range(100)])
    for _ in range(2):
        result, = intake.poll()
        assert result.reason_code == 'SOURCE_ITEMS_HELD'
        assert len(result.item_holds) == 100
        assert {code for _, code in result.item_holds} == {'SOURCE_ITEM_RETAIN_FAILED'}
        _, data = events[-1]
        assert data['failure_count'] == 100 and data['overflow_count'] == 0
        assert len(data['groups']) == 1
        assert data['groups'][0]['count'] == 100
        assert data['groups'][0]['stage'] == 'MANUAL_SECTION'
        assert data['groups'][0]['exception_class'] == 'ValueError'
        assert data['groups'][0]['file'] == 'test_native_source_failure_summary.py'
        assert data['groups'][0]['function'] == '_failed'
        assert data['groups'][0]['line'] > 0
    assert len(events) == 2
    assert 'TOKEN' not in json.dumps(events) and 'source-body' not in json.dumps(events)


@pytest.mark.parametrize('stage', ('ROOT_POLL', 'ITEM', 'DECLARED_CHILD', 'DECLARED_ASSET'))
def test_actual_catch_stages_preserve_hold_and_expose_only_safe_frame(monkeypatch, stage):
    intake, events = _intake(monkeypatch, 'UK-01')
    error = AuthorityPersistenceError('private body TOKEN=hidden')
    item = SourceItem('UK-01', 'key', 'Title', 'Body', 'https://www.gov.uk/item')
    if stage == 'ROOT_POLL':
        intake._poll_one = lambda _: (_ for _ in ()).throw(error)
    elif stage == 'ITEM':
        intake._definitions = {'UK-01': 'definition'}
        intake._sources = NS(current_summary=lambda *a, **k: NS(version_id='version'),
            version_details=lambda *a, **k: NS(request=NS(locator=n.SOURCE_URLS['UK-01'])))
        intake._proof = None
        intake._licence = NS(for_source=lambda **_: NS(decision='PERMITTED', record_id='rights'))
        monkeypatch.setattr(n, 'parse_observation', lambda **_: (item,))
        intake._fetch_complete_item = lambda *a: (_ for _ in ()).throw(error)
    else:
        if stage == 'DECLARED_CHILD':
            hold = n.NativeSourceIntakeHold('DECLARED_PARENT', child_items=(('/item-child', 'Child'),))
            intake._fetch_manual_sections = lambda *a: ((item, _failed(error)),)
            intake._fetch_declared_assets = lambda *a: ()
        else:
            asset = 'https://assets.publishing.service.gov.uk/media/fixture/table.csv'
            hold = n.NativeSourceIntakeHold('DECLARED_PARENT', unsupported_attachments=((asset, 'Table'),))
            intake._fetch_manual_sections = lambda *a: ()
            monkeypatch.setattr(n, 'declared_spreadsheet', lambda *a, **k: NS(asset_url=asset))
            intake._fetch_declared_assets = lambda *a: ((NS(asset_url=asset), _failed(error)),)
        intake._parse_complete_item = lambda *a: (_ for _ in ()).throw(hold)
        def poll_one(source):
            units, observations, holds = intake._settle_item(source, 'definition', 'version', NS(),
                item, 'https://www.gov.uk/api/content/item', b'{}', NOW, 'rights')
            return n.NativeSourceDisposition(source, 'HOLD', 'SOURCE_ITEMS_HELD', units, observations=observations, item_holds=holds)
        intake._poll_one = poll_one
    result, = intake.poll()
    assert result.status == 'HOLD'
    assert len(events) == 1
    _, data = events[0]
    assert data['groups'][0]['stage'] == stage
    assert data['groups'][0]['exception_class'] == 'AuthorityPersistenceError'
    assert 'TOKEN' not in json.dumps(data) and 'private body' not in json.dumps(data)


def test_transport_cause_and_authority_are_distinct_and_key_overflow_is_bounded(monkeypatch):
    intake, events = _intake(monkeypatch)
    transport = ValueError('transport secret')
    transport.__cause__ = urllib.error.URLError('https://secret.invalid/')
    errors = [transport, AuthorityPersistenceError('authority secret')]
    errors += [type(f'Failure{i}', (ValueError,), {})('hidden') for i in range(20)]
    _manual(intake, monkeypatch, errors)
    intake.poll()
    _, data = events[0]
    assert data['failure_count'] == 22 and len(data['groups']) == 16
    assert data['overflow_count'] == 6
    assert data['groups'][0]['cause_class'] == 'URLError'
    assert data['groups'][1]['exception_class'] == 'AuthorityPersistenceError'
    assert all(set(group) == {'stage','exception_class','file','function','line','cause_class','count'} for group in data['groups'])


@pytest.mark.parametrize('stopped', (False, True))
def test_diagnostic_failure_never_replaces_hold_or_owner_stop(monkeypatch, stopped):
    intake, _ = _intake(monkeypatch)
    errors = [ValueError('original failure')]
    if stopped:
        errors.append(VetoError('signed stop'))
    _manual(intake, monkeypatch, errors)
    monkeypatch.setattr(n, 'emit_diagnostic', lambda *a: (_ for _ in ()).throw(OSError('logger failed')))
    if stopped:
        with pytest.raises(VetoError, match='signed stop'):
            intake.poll()
    else:
        assert intake.poll()[0].item_holds[0][1] == 'SOURCE_ITEM_RETAIN_FAILED'
    assert intake._poll_failure_groups is intake._poll_failure_overflow is None


def test_structured_parser_hold_keeps_its_original_reason_code(monkeypatch):
    intake, events = _intake(monkeypatch)
    _manual(intake, monkeypatch, [n.NativeSourceIntakeHold('SOURCE_ITEM_METADATA_HOLD')])
    result, = intake.poll()
    assert result.item_holds[0][1] == 'SOURCE_ITEM_METADATA_HOLD'
    assert events[0][1]['groups'][0]['exception_class'] == 'NativeSourceIntakeHold'


def test_non_code_traceback_labels_do_not_emit_url_or_data(monkeypatch):
    intake, events = _intake(monkeypatch)
    scope = {}
    exec(compile('def fail():\n    raise ValueError("private body")',
        'https://secret.invalid/TOKEN-hidden', 'exec'), scope)
    future = Future()
    try:
        scope['fail']()
    except ValueError as error:
        future.set_exception(error)
    monkeypatch.setattr(n, 'parse_govuk_manual_inventory', lambda *a, **k: NS(sections=(('/item', 'Item'),)))
    intake._fetch_manual_sections = lambda *a: ((SourceItem('UK-03','key','Title','Body','https://www.gov.uk/item'), future),)
    intake._poll_one = lambda source: intake._poll_direct_govuk(source,'definition','version',NS(),NS(record_id='rights'))
    intake.poll()
    assert events[0][1]['groups'][0]['file'] == 'OTHER'
    assert 'TOKEN' not in json.dumps(events)
