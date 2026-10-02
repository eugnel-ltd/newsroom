"""Durable native continuation state, independent of expiring diagnostic payloads."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes

VERSION = 'newsroom.native-current.v1'
TABLES = ('native_current_sources', 'native_current_heads', 'native_current_pairs',
          'native_current_observations', 'native_current_portfolio', 'native_embedding_progress_pins')
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS native_current_meta(singleton INTEGER PRIMARY KEY CHECK(singleton=1), version TEXT NOT NULL, counts_json TEXT NOT NULL, digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_sources(revision_id TEXT PRIMARY KEY, land_seq INTEGER NOT NULL UNIQUE, land_digest TEXT NOT NULL, content_json TEXT NOT NULL, content_digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_pairs(pair_digest TEXT PRIMARY KEY, pair_json TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_heads(revision_id TEXT PRIMARY KEY REFERENCES native_current_sources(revision_id), ordinal INTEGER NOT NULL CHECK(ordinal>0), diagnostic_seq INTEGER NOT NULL, diagnostic_digest TEXT NOT NULL, state_json TEXT NOT NULL, state_digest TEXT NOT NULL, pair_digest TEXT REFERENCES native_current_pairs(pair_digest))",
    "CREATE INDEX IF NOT EXISTS native_current_heads_pair ON native_current_heads(pair_digest)",
    "CREATE TABLE IF NOT EXISTS native_current_observations(observation_digest TEXT PRIMARY KEY, reference_json TEXT NOT NULL, reference_digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_portfolio(singleton INTEGER PRIMARY KEY CHECK(singleton=1), diagnostic_seq INTEGER NOT NULL, diagnostic_digest TEXT NOT NULL, portfolio_json TEXT NOT NULL, portfolio_digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_units(ingest_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL REFERENCES native_current_sources(revision_id), effective_revision_digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_embedding_progress_pins(passage_id TEXT NOT NULL, cycle_id TEXT NOT NULL, progress_seq INTEGER NOT NULL REFERENCES ledger(seq), unit_ingest_id TEXT NOT NULL, revision_id TEXT NOT NULL REFERENCES native_current_sources(revision_id), progress_digest TEXT NOT NULL, PRIMARY KEY(passage_id,cycle_id,progress_seq,unit_ingest_id))",
)


def _json(value: object) -> str:
    return canonical_json_bytes(value).decode()


def checked_json(raw: str, digest: str, *, label: str) -> dict:
    if type(raw) is not str or digest_bytes(raw.encode()) != digest:
        raise ValueError(f'native CURRENT {label} payload differs')
    value = json.loads(raw)
    if type(value) is not dict or _json(value) != raw:
        raise ValueError(f'native CURRENT {label} is not canonical')
    return value


def ensure_schema(connection: sqlite3.Connection, *, initialise_empty: bool = False) -> None:
    for statement in SCHEMA:
        connection.execute(statement)
    if initialise_empty:
        write_counts(connection)
    connection.commit()


def write_counts(connection: sqlite3.Connection) -> None:
    counts = {table: connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
              for table in (*TABLES, 'native_current_units')}
    value = {'version': VERSION, 'counts': counts}
    connection.execute('INSERT OR REPLACE INTO native_current_meta VALUES(1,?,?,?)',
                       (VERSION, _json(counts), digest_bytes(canonical_json_bytes(value))))


def require_ready(connection: sqlite3.Connection, *, verify_inventory: bool = True) -> dict:
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_current_meta'").fetchone():
        raise ValueError('native CURRENT state requires explicit legacy import')
    row = connection.execute('SELECT version,counts_json,digest FROM native_current_meta WHERE singleton=1').fetchone()
    if row is None:
        raise ValueError('native CURRENT state requires explicit legacy import')
    version, raw, digest = row
    counts = json.loads(raw)
    if version != VERSION or _json(counts) != raw or digest_bytes(canonical_json_bytes({'version': version, 'counts': counts})) != digest:
        raise ValueError('native CURRENT readiness differs')
    if set(counts) != set((*TABLES, 'native_current_units')):
        raise ValueError('native CURRENT inventory differs')
    if not verify_inventory:
        return counts
    for table, expected in counts.items():
        if type(expected) is not int or expected < 0 or connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] != expected:
            raise ValueError('native CURRENT inventory differs')
    if connection.execute('SELECT 1 FROM native_current_heads h LEFT JOIN native_current_sources s ON s.revision_id=h.revision_id LEFT JOIN native_current_pairs p ON p.pair_digest=h.pair_digest WHERE s.revision_id IS NULL OR (h.pair_digest IS NOT NULL AND p.pair_digest IS NULL) LIMIT 1').fetchone():
        raise ValueError('native CURRENT root binding differs')
    return counts


def current_state_retained_sequences(connection: sqlite3.Connection) -> frozenset[int]:
    """Exact immutable business/diagnostic roots; never read diagnostic bodies."""
    require_ready(connection)
    return frozenset(row[0] for row in connection.execute(
        "SELECT land_seq FROM native_current_sources UNION SELECT diagnostic_seq FROM native_current_heads "
        "WHERE json_extract(state_json,'$.stage') IN ('EMBEDDING_STARTED','ASSESSMENT_STARTED','PUBLICATION_STARTED','ACKNOWLEDGED','COPY_CORRECTION_PREPARED') "
        'UNION SELECT diagnostic_seq FROM native_current_portfolio UNION SELECT progress_seq FROM native_embedding_progress_pins') if row[0] > 0)


def _source_encoding(value: dict) -> dict:
    value = json.loads(_json(value))
    if len(value['units']) > 1 and 'shared_body' not in value:
        bodies = [item['body'] for item in value['units']]
        if any(body != bodies[0] for body in bodies):
            raise ValueError('native progress shared body differs')
        value['shared_body'] = bodies[0]
        for item in value['units']:
            del item['body']
    return value


def retain_source(connection: sqlite3.Connection, value: dict, *, seq: int, digest: str, units: tuple) -> None:
    # Preserve logical units without retaining one full body per chunk.
    value = _source_encoding(value)
    raw = _json(value)
    connection.execute('INSERT INTO native_current_sources VALUES(?,?,?,?,?)',
                       (value['revision_id'], seq, digest, raw, digest_bytes(raw.encode())))
    connection.executemany('INSERT INTO native_current_units VALUES(?,?,?)',
                          ((unit.ingest_id, unit.revision_id, digest_bytes(canonical_json_bytes(asdict(unit.effective_revision)))) for unit in units))


def retain_head(connection: sqlite3.Connection, logical: dict, *, seq: int, digest: str, pair_digest: str | None, expected_previous_ordinal: int | None = None) -> None:
    previous = connection.execute('SELECT pair_digest,ordinal,state_json,state_digest FROM native_current_heads WHERE revision_id=?', (logical['revision_id'],)).fetchone()
    if expected_previous_ordinal is not None:
        if (0 if previous is None else previous[1]) != expected_previous_ordinal:
            raise ValueError('native CURRENT writer ordinal differs')
        if previous is not None:
            prior = json.loads(previous[2])
            if (_json(prior) != previous[2] or prior.get('revision_id') != logical['revision_id']
                    or prior.get('ordinal') != previous[1]
                    or digest_bytes(canonical_json_bytes({'state': prior, 'pair_digest': previous[0]})) != previous[3]):
                raise ValueError('native CURRENT prior head payload differs')
    facts = dict(logical['facts'])
    if pair_digest is not None:
        pair = {key: facts.pop(key) for key in ('retrieval_binding', 'retrieval_rights_inventory')}
        raw = _json(pair)
        if digest_bytes(raw.encode()) != pair_digest:
            raise ValueError('native CURRENT pair binding differs')
        connection.execute('INSERT OR IGNORE INTO native_current_pairs VALUES(?,?)', (pair_digest, raw))
        prior = connection.execute('SELECT pair_json FROM native_current_pairs WHERE pair_digest=?', (pair_digest,)).fetchone()
        if prior is None or prior[0] != raw:
            raise ValueError('native CURRENT pair payload differs')
    value = {**logical, 'facts': facts}
    raw = _json(value)
    bound = {'state': value, 'pair_digest': pair_digest}
    connection.execute('INSERT OR REPLACE INTO native_current_heads VALUES(?,?,?,?,?,?,?)',
                       (logical['revision_id'], logical['ordinal'], seq, digest, raw,
                        digest_bytes(canonical_json_bytes(bound)), pair_digest))
    if previous and previous[0] is not None and previous[0] != pair_digest:
        connection.execute('DELETE FROM native_current_pairs WHERE pair_digest=? AND NOT EXISTS(SELECT 1 FROM native_current_heads WHERE pair_digest=?)', (previous[0], previous[0]))


def retain_portfolio(connection: sqlite3.Connection, value: dict, *, seq: int, digest: str) -> None:
    raw = _json(value)
    connection.execute('INSERT OR REPLACE INTO native_current_portfolio VALUES(1,?,?,?,?)',
                       (seq, digest, raw, digest_bytes(raw.encode())))
    for source in value['sources']:
        for reference in source.get('observations', ()):
            if len(reference) != 4 or any(type(item) is not str or not item for item in reference):
                raise ValueError('native source observation reference differs')
            raw = _json({'reference': list(reference)})
            connection.execute('INSERT OR IGNORE INTO native_current_observations VALUES(?,?,?)',
                               (reference[1], raw, digest_bytes(raw.encode())))


def retain_embedding_pins(connection: sqlite3.Connection, value: dict, *, seq: int, digest: str) -> None:
    if value.get('stage') != 'EMBEDDING_STARTED':
        return
    for unit_id, retained in value['facts'].get('retrieval_embeddings', {}).items():
        if isinstance(retained, dict) and retained.get('state') == 'STARTED':
            connection.execute('INSERT OR IGNORE INTO native_embedding_progress_pins VALUES(?,?,?,?,?,?)',
                               (retained['passage_id'], retained['cycle_id'], seq, unit_id, value['revision_id'], digest))


def selected_landing_rows(connection: sqlite3.Connection, *, ingest_id: str, revision_id_hint: str | None = None, effective_revision_digest: str | None = None) -> list[tuple]:
    """Authenticate selected source content and its original immutable LAND pin."""
    require_ready(connection, verify_inventory=False)
    rows = connection.execute('SELECT s.revision_id,s.land_seq,s.land_digest,s.content_json,s.content_digest,u.effective_revision_digest FROM native_current_units u JOIN native_current_sources s ON s.revision_id=u.revision_id WHERE u.ingest_id=?', (ingest_id,)).fetchall()
    result = []
    for revision, seq, digest, raw, content_digest, effective in rows:
        if revision_id_hint is not None and revision != revision_id_hint or effective_revision_digest is not None and effective != effective_revision_digest:
            continue
        value = checked_json(raw, content_digest, label='source')
        if value.get('revision_id') != revision:
            raise ValueError('native CURRENT source binding differs')
        pin = connection.execute("SELECT payload_digest,payload_json FROM ledger WHERE seq=? AND kind='NATIVE_REVISION_LANDED'", (seq,)).fetchone()
        if pin is None or pin[0] != digest:
            raise ValueError('native CURRENT source LAND pin differs')
        pinned = checked_json(pin[1], digest, label='source LAND pin')
        if _source_encoding(pinned) != value:
            raise ValueError('native CURRENT source LAND content differs')
        # LAND is immutable business evidence, not an expiring debug stream.
        # Preserve selected-revision conflicting-LAND denial through the existing
        # partial index; never replay unrelated source/progress history.
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='model_usage_native_landed_revision'").fetchone():
            for other_digest, other_raw in connection.execute(
                "SELECT payload_digest,payload_json FROM ledger WHERE json_extract(payload_json,'$.revision_id')=? AND kind='NATIVE_REVISION_LANDED'", (revision,)):
                other = checked_json(other_raw, other_digest, label='source LAND pin')
                if _source_encoding(other) != value:
                    raise ValueError('native conservative source landing changed')
        result.append(pin)
    return result
