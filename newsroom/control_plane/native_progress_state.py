"""Durable native continuation state, independent of expiring diagnostic payloads."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes

VERSION = 'newsroom.native-current.v1'
PAIR_ITEMS_VERSION = 'newsroom.native-pair-items.v1'
_ITEM_TABLE = 'native_current_inventory_items'
_PAIR_ITEM_TABLES = (_ITEM_TABLE, 'native_current_pair_inventories', 'native_current_pair_items')
_MAX_PAIR_ITEMS = 4096
TABLES = ('native_current_sources', 'native_current_heads', 'native_current_pairs',
          'native_current_observations', 'native_current_portfolio', 'native_embedding_progress_pins',
          *_PAIR_ITEM_TABLES)
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS native_current_meta(singleton INTEGER PRIMARY KEY CHECK(singleton=1), version TEXT NOT NULL, counts_json TEXT NOT NULL, digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_sources(revision_id TEXT PRIMARY KEY, land_seq INTEGER NOT NULL UNIQUE, land_digest TEXT NOT NULL, content_json TEXT NOT NULL, content_digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_pairs(pair_digest TEXT PRIMARY KEY, pair_json TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_inventory_items(item_id INTEGER PRIMARY KEY, item_digest TEXT NOT NULL UNIQUE, item_json TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS native_current_pair_inventories(inventory_id INTEGER PRIMARY KEY, pair_digest TEXT NOT NULL UNIQUE REFERENCES native_current_pairs(pair_digest) ON DELETE CASCADE)",
    "CREATE TABLE IF NOT EXISTS native_current_pair_items(inventory_id INTEGER NOT NULL REFERENCES native_current_pair_inventories(inventory_id) ON DELETE CASCADE, item_id INTEGER NOT NULL REFERENCES native_current_inventory_items(item_id), PRIMARY KEY(inventory_id,item_id)) WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS native_current_pair_item_consumers ON native_current_pair_items(item_id)",
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
    expected_tables = set((*TABLES, 'native_current_units'))
    if set(counts) not in (expected_tables, expected_tables - set(_PAIR_ITEM_TABLES)):
        raise ValueError('native CURRENT inventory differs')
    if set(counts) == expected_tables - set(_PAIR_ITEM_TABLES):
        for table in _PAIR_ITEM_TABLES:
            present = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if present and connection.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone():
                raise ValueError('native CURRENT uncounted pair items differ')
    if not verify_inventory:
        return counts
    for table, expected in counts.items():
        if type(expected) is not int or expected < 0 or connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] != expected:
            raise ValueError('native CURRENT inventory differs')
    if connection.execute('SELECT 1 FROM native_current_heads h LEFT JOIN native_current_sources s ON s.revision_id=h.revision_id LEFT JOIN native_current_pairs p ON p.pair_digest=h.pair_digest WHERE s.revision_id IS NULL OR (h.pair_digest IS NOT NULL AND p.pair_digest IS NULL) LIMIT 1').fetchone():
        raise ValueError('native CURRENT root binding differs')
    return counts


def read_pair(connection: sqlite3.Connection, pair_digest: str) -> dict:
    """Reconstruct exact logical bytes through bounded selected item/root indexes."""
    row = connection.execute('SELECT pair_json FROM native_current_pairs WHERE pair_digest=?', (pair_digest,)).fetchone()
    if row is None:
        raise ValueError('native CURRENT retrieval pair root is missing')
    raw = row[0]
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError('native CURRENT retrieval pair payload differs') from exc
    if type(value) is not dict or _json(value) != raw:
        raise ValueError('native CURRENT retrieval pair is not canonical')
    if set(value) == {'retrieval_binding', 'retrieval_rights_inventory'}:
        pair = value
    else:
        if (set(value) != {'schema', 'retrieval_binding', 'inventory_items'}
                or value['schema'] != PAIR_ITEMS_VERSION or type(value['inventory_items']) is not list
                or len(value['inventory_items']) > _MAX_PAIR_ITEMS
                or any(type(item) is not int or item < 1 for item in value['inventory_items'])):
            raise ValueError('native CURRENT pair item format differs')
        root = connection.execute('SELECT inventory_id FROM native_current_pair_inventories WHERE pair_digest=?', (pair_digest,)).fetchone()
        if root is None:
            raise ValueError('native CURRENT pair inventory root is missing')
        members = {row[0] for row in connection.execute('SELECT item_id FROM native_current_pair_items WHERE inventory_id=?', (root[0],))}
        if members != set(value['inventory_items']):
            raise ValueError('native CURRENT pair inventory references differ')
        rows = connection.execute(
            'SELECT refs.key,items.item_digest,items.item_json FROM json_each(?) refs '
            'JOIN native_current_inventory_items items ON items.item_id=refs.value '
            'ORDER BY CAST(refs.key AS INTEGER)', (_json(value['inventory_items']),),
        ).fetchall()
        if len(rows) != len(value['inventory_items']) or any(index != row[0] for index, row in enumerate(rows)):
            raise ValueError('native CURRENT pair inventory item is missing')
        pair = {'retrieval_binding': value['retrieval_binding'], 'retrieval_rights_inventory': [
            checked_json(item_raw, item_digest, label='inventory item') for _index, item_digest, item_raw in rows]}
    if digest_bytes(canonical_json_bytes(pair)) != pair_digest:
        raise ValueError('native CURRENT retrieval pair payload differs')
    return pair


def _item_encoding(connection: sqlite3.Connection, pair: dict, pair_digest: str) -> str:
    rows = [(digest_bytes(canonical_json_bytes(item)), _json(item)) for item in pair['retrieval_rights_inventory']]
    connection.executemany('INSERT OR IGNORE INTO native_current_inventory_items(item_digest,item_json) VALUES(?,?)', rows)
    ids = connection.execute('SELECT refs.key,item_id FROM json_each(?) refs JOIN native_current_inventory_items ON item_digest=refs.value ORDER BY CAST(refs.key AS INTEGER)',
                             (_json([digest for digest, _raw in rows]),)).fetchall()
    if len(ids) != len(rows):
        raise ValueError('native CURRENT item retention differs')
    connection.execute('INSERT OR IGNORE INTO native_current_pair_inventories(pair_digest) VALUES(?)', (pair_digest,))
    inventory = connection.execute('SELECT inventory_id FROM native_current_pair_inventories WHERE pair_digest=?', (pair_digest,)).fetchone()[0]
    connection.executemany('INSERT OR IGNORE INTO native_current_pair_items VALUES(?,?)', ((inventory, item_id) for _index, item_id in ids))
    return _json({'schema': PAIR_ITEMS_VERSION, 'retrieval_binding': pair['retrieval_binding'],
                  'inventory_items': [item_id for _index, item_id in ids]})


def retain_pair(connection: sqlite3.Connection, pair: dict, pair_digest: str) -> None:
    if set(pair) != {'retrieval_binding', 'retrieval_rights_inventory'} or digest_bytes(canonical_json_bytes(pair)) != pair_digest:
        raise ValueError('native CURRENT pair binding differs')
    if connection.execute('SELECT 1 FROM native_current_pairs WHERE pair_digest=?', (pair_digest,)).fetchone():
        if read_pair(connection, pair_digest) != pair:
            raise ValueError('native CURRENT pair payload differs')
        return
    # Preserve previously admitted non-inventory JSON shapes without a new gate.
    raw = _json(pair)
    connection.execute('INSERT INTO native_current_pairs VALUES(?,?)', (pair_digest, raw))
    if type(pair['retrieval_rights_inventory']) is list and len(pair['retrieval_rights_inventory']) <= _MAX_PAIR_ITEMS and all(type(item) is dict for item in pair['retrieval_rights_inventory']):
        encoded = _item_encoding(connection, pair, pair_digest)
        connection.execute('UPDATE native_current_pairs SET pair_json=? WHERE pair_digest=?', (encoded, pair_digest))
    if read_pair(connection, pair_digest) != pair:
        raise ValueError('native CURRENT retained inventory item differs')


def _remove_pair(connection: sqlite3.Connection, pair_digest: str) -> None:
    if connection.execute('SELECT 1 FROM native_current_heads WHERE pair_digest=?', (pair_digest,)).fetchone():
        return
    read_pair(connection, pair_digest)
    candidates = [row[0] for row in connection.execute('SELECT i.item_id FROM native_current_pair_items i JOIN native_current_pair_inventories p USING(inventory_id) WHERE p.pair_digest=?', (pair_digest,))]
    connection.execute('DELETE FROM native_current_pairs WHERE pair_digest=?', (pair_digest,))
    connection.executemany('DELETE FROM native_current_inventory_items WHERE item_id=? AND NOT EXISTS(SELECT 1 FROM native_current_pair_items WHERE item_id=?)', ((item_id, item_id) for item_id in candidates))


def compact_current_pairs(connection: sqlite3.Connection) -> dict:
    """Explicit atomic conversion preserving referenced pair PKs and logical bytes."""
    if connection.in_transaction:
        raise ValueError('native pair compaction requires its own transaction')
    connection.execute('BEGIN IMMEDIATE')
    converted = 0
    try:
        require_ready(connection)
        digests = tuple(row[0] for row in connection.execute('SELECT pair_digest FROM native_current_pairs'))
        for digest in digests:
            pair = read_pair(connection, digest)
            raw = connection.execute('SELECT pair_json FROM native_current_pairs WHERE pair_digest=?', (digest,)).fetchone()[0]
            inventory = pair['retrieval_rights_inventory']
            if set(json.loads(raw)) == {'retrieval_binding', 'retrieval_rights_inventory'} and type(inventory) is list and len(inventory) <= _MAX_PAIR_ITEMS and all(type(item) is dict for item in inventory):
                encoded = _item_encoding(connection, pair, digest)
                connection.execute('UPDATE native_current_pairs SET pair_json=? WHERE pair_digest=?', (encoded, digest))
                if read_pair(connection, digest) != pair:
                    raise ValueError('native CURRENT converted pair differs')
                converted += 1
        write_counts(connection)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return {'converted_pairs': converted, 'inventory_items': connection.execute('SELECT count(*) FROM native_current_inventory_items').fetchone()[0]}


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
        retain_pair(connection, pair, pair_digest)
    value = {**logical, 'facts': facts}
    raw = _json(value)
    bound = {'state': value, 'pair_digest': pair_digest}
    connection.execute('INSERT OR REPLACE INTO native_current_heads VALUES(?,?,?,?,?,?,?)',
                       (logical['revision_id'], logical['ordinal'], seq, digest, raw,
                        digest_bytes(canonical_json_bytes(bound)), pair_digest))
    if previous and previous[0] is not None and previous[0] != pair_digest:
        _remove_pair(connection, previous[0])


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
