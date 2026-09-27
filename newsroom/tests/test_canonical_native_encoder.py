"""The native encoder changes cost, not the restricted canonical authority."""
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import orjson
import pytest

from newsroom.tests.test_canonical_fast_validation import _outcome


def test_validated_exact_builtins_use_native_bytes_once(monkeypatch):
    calls = []
    original = orjson.dumps

    def counted(value, **kwargs):
        calls.append(kwargs['option'])
        return original(value, **kwargs)

    monkeypatch.setattr(orjson, 'dumps', counted)
    value = {'香港': ['🙂', -9007199254740991, 9007199254740991], 'a': (True, None)}
    assert _outcome(value) == _outcome(value, diagnostic=True)
    assert calls == [orjson.OPT_SORT_KEYS | orjson.OPT_STRICT_INTEGER]


def test_custom_and_invalid_values_never_enter_native_encoder(monkeypatch):
    class CustomDict(dict):
        pass

    class CustomString(str):
        pass

    @dataclass
    class Record:
        name: str

    def unexpected(*_args, **_kwargs):
        pytest.fail('non-builtin or invalid value reached native encoder')

    monkeypatch.setattr(orjson, 'dumps', unexpected)
    for value in (
        CustomDict(a=1), {'a': CustomString('香港')}, {'bad': [1.5]},
        {'bad': [float('nan')]}, {'bad': [2**53]}, {'bad': b'bytes'},
        {'bad': '\ud800'}, {1: 'key'}, datetime(2026, 1, 1, tzinfo=UTC),
        UUID(int=0), Record('not an admitted JSON type'),
    ):
        assert _outcome(value) == _outcome(value, diagnostic=True)


def test_encoder_depth_limit_preserves_stdlib_fallback(monkeypatch):
    original = orjson.dumps
    attempts = []

    def counted(value, **kwargs):
        attempts.append(True)
        return original(value, **kwargs)

    monkeypatch.setattr(orjson, 'dumps', counted)
    value = 'leaf'
    for _ in range(300):
        value = [value]
    assert _outcome(value) == _outcome(value, diagnostic=True)
    assert attempts == [True]


def test_every_unicode_scalar_encodes_to_identical_bytes():
    value = ''.join(chr(n) for n in range(0x110000) if not 0xD800 <= n <= 0xDFFF)
    assert _outcome(value) == _outcome(value, diagnostic=True)


def test_control_prefix_and_unicode_key_order_is_unchanged():
    keys = ['', 'a', 'A', 'ä', '香港', '🙂', '\ud7ff', '\ue000', '\U0010ffff']
    keys.extend(chr(n) for n in range(128))
    keys.extend('prefix' + chr(n) for n in range(0, 0x110000, 1024) if not 0xD800 <= n <= 0xDFFF)
    value = {key: i for i, key in enumerate(reversed(keys))}
    assert _outcome(value) == _outcome(value, diagnostic=True)
