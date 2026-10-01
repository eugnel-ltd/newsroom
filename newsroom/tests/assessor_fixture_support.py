"""Reuse an immutable candidate value, never an assessor usage database."""

from functools import lru_cache
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from newsroom.tests.test_increment10_ingress import _candidate


@lru_cache(maxsize=1)
def candidate_value():
    with TemporaryDirectory(prefix="newsroom-assessor-candidate-") as directory:
        connection, _unused_port, candidate = _candidate(Path(directory))
        try:
            return candidate
        finally:
            connection.close()


def candidate_fixture(_tmp_path):
    # Existing callers only close this unused handle; usage stores stay fresh.
    candidate = candidate_value()
    return sqlite3.connect(":memory:"), None, candidate
