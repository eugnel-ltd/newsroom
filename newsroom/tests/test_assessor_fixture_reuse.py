"""Assessor-only value reuse never retains or shares candidate-store authority."""

import sqlite3
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import pytest

from newsroom.increment10.evidence import _base_package
from newsroom.increment6.candidates import StoryCandidateVersion
from newsroom.tests import assessor_fixture_support as support
from newsroom.tests.test_increment10_editorial import _ready_package


@pytest.fixture(autouse=True)
def _isolated_candidate_cache():
    support.candidate_value.cache_clear()
    try:
        yield
    finally:
        support.candidate_value.cache_clear()


def _assert_immutable(value):
    if is_dataclass(value):
        assert value.__dataclass_params__.frozen
        for field in fields(value):
            _assert_immutable(getattr(value, field.name))
    elif type(value) is tuple:
        for item in value:
            _assert_immutable(item)
    else:
        assert isinstance(value, (str, bytes, int, bool, type(None)))


def test_one_real_builder_reuses_only_value_and_closes_authority(tmp_path, monkeypatch):
    real_builder = support._candidate
    builds = []

    def tracked_builder(path):
        connection, port, candidate = real_builder(path)
        builds.append((path, connection, candidate.canonical_bytes,
                       candidate.governing_manifest.canonical_bytes))
        return connection, port, candidate

    monkeypatch.setattr(support, "_candidate", tracked_builder)
    first, first_port, candidate = support.candidate_fixture(tmp_path / "first")
    second, second_port, repeated = support.candidate_fixture(tmp_path / "second")
    try:
        assert support.candidate_value() is candidate is repeated
        assert support.candidate_value.cache_parameters() == {"maxsize": 1, "typed": False}
        assert support.candidate_value.cache_info().misses == len(builds) == 1
        path, authority, canonical_bytes, manifest_bytes = builds[0]
        assert not path.exists()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            authority.execute("SELECT 1")
        assert type(candidate) is StoryCandidateVersion
        assert candidate.canonical_bytes == canonical_bytes
        assert candidate.governing_manifest.canonical_bytes == manifest_bytes
        assert first_port is second_port is None
        assert first is not second
        for connection in (first, second):
            assert connection.execute("PRAGMA database_list").fetchall() == [(0, "main", "")]
            assert connection.execute("SELECT count(*) FROM sqlite_master").fetchone() == (0,)
        first.execute("CREATE TABLE private_state (value TEXT)")
        first.execute("INSERT INTO private_state VALUES ('first only')")
        assert second.execute("SELECT name FROM sqlite_master").fetchall() == []
        first.close()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            first.execute("SELECT 1")
        assert second.execute("SELECT 1").fetchone() == (1,)
        assert list(tmp_path.iterdir()) == []
    finally:
        first.close()
        second.close()


def test_frozen_candidate_and_independent_package_variants_preserve_cached_value():
    candidate = support.candidate_value()
    canonical_bytes = candidate.canonical_bytes
    manifest_bytes = candidate.governing_manifest.canonical_bytes
    _assert_immutable(candidate)
    with pytest.raises(FrozenInstanceError):
        candidate.ordinal = 2
    with pytest.raises(FrozenInstanceError):
        candidate.governing_manifest.proposed_summary = "changed"
    first = _base_package(_ready_package(candidate)[1])
    second = _base_package(_ready_package(candidate)[1])
    assert first is not second
    assert first.digest == second.digest
    variant = replace(first, passages=("A separate package variant.",))
    assert variant.digest != second.digest
    assert first.passages == second.passages
    exported = candidate.canonical_value
    exported["governing_manifest"]["proposed_summary"] = "changed copy"
    assert support.candidate_value() is candidate
    assert candidate.canonical_bytes == canonical_bytes
    assert candidate.governing_manifest.canonical_bytes == manifest_bytes
    assert _base_package(_ready_package(candidate)[1]).digest == second.digest
