"""Native continuation from durable CURRENT rows, not diagnostic-history replay.

The explicit legacy importer alone reads old diagnostics. Cold retrieval pairs
remain content-addressed and are authenticated only when selected. Business
provider pins remain immutable ledger records; ordinary logs are compact DTOs.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.effective_revision import EffectiveRevisionIdentity

from .corpus import CorpusAuthorityBinding, CorpusIngestUnit
from .store import append_ledger

LAND = "NATIVE_REVISION_LANDED"
STATE = "NATIVE_REVISION_PROGRESS"
PORTFOLIO = "NATIVE_SOURCE_PORTFOLIO"

_RETRIEVAL_FIELDS = ("retrieval_binding", "retrieval_rights_inventory")


def _pair_digest(facts: dict) -> str | None:
    if not all(key in facts for key in _RETRIEVAL_FIELDS):
        return None
    return digest_bytes(canonical_json_bytes({key: facts[key] for key in _RETRIEVAL_FIELDS}))


def _state_digest(stage: str, facts: dict, pair_digest: str | None) -> str:
    # The verified pair digest avoids re-serialising large referenced values on
    # replay. Keep every other fact inline, including embedding evidence.
    remaining = {key: value for key, value in facts.items()
                 if pair_digest is None or key not in _RETRIEVAL_FIELDS}
    return digest_bytes(canonical_json_bytes(
        {"stage": stage, "facts": remaining, "retrieval_pair_digest": pair_digest}
    ))


def _share_progress_strings(progress: dict) -> None:
    """Share equal immutable text after replay, never lists/dicts or a persistent cache."""
    strings: dict[str, str] = {}

    def visit(value):
        if type(value) is str:
            return strings.setdefault(value, value)
        if type(value) is dict:
            # Preserve insertion order as well as values: iteration can order
            # continuation work. Each original mutable container keeps its identity.
            replacement = {strings.setdefault(key, key): visit(item)
                           for key, item in value.items()}
            value.clear()
            value.update(replacement)
        elif type(value) is list:
            for index, item in enumerate(value):
                value[index] = visit(item)
        return value

    try:
        visit(progress)
    finally:
        # The recursive closure may await cyclic GC; do not leave its pool alive.
        strings.clear()


@dataclass(frozen=True, slots=True)
class _ProgressRecord:
    seq: int
    payload_digest: str
    ordinal: int
    pair_digest: str | None
    state_digest: str
    pair_root: tuple[int, str, int] | None

    def reference(self) -> dict:
        return {"seq": self.seq, "payload_digest": self.payload_digest,
                "ordinal": self.ordinal}


def _unit(value: dict, bodies: dict[str, str]) -> CorpusIngestUnit:
    value = dict(value)
    value["body"] = bodies.setdefault(value["body"], value["body"])
    value["effective_revision"] = EffectiveRevisionIdentity(**value["effective_revision"])
    authority = dict(value["authority"])
    authority["records"] = tuple(authority["records"])
    value["authority"] = CorpusAuthorityBinding(**authority)
    return CorpusIngestUnit(**value)


def _landed_units(
    value: dict, *, bodies: dict[str, str] | None = None,
) -> tuple[CorpusIngestUnit, ...]:
    """Decode both retained encodings for journal and selected accounting reads."""
    bodies = {} if bodies is None else bodies
    raw_units = value.get("units", ())
    if len(raw_units) > 1:
        # Chunk authority/identity text repeats too. Reuse replay's immutable
        # sharing without aliasing any individual receipt's lists or dicts.
        _share_progress_strings(value)
    if "shared_body" in value:
        body = value["shared_body"]
        if type(body) is not str or any("body" in item for item in raw_units):
            raise ValueError("native progress shared body differs")
        raw_units = ({**item, "body": body} for item in raw_units)
    return tuple(_unit(item, bodies) for item in raw_units)


class NativeRevisionJournal:
    def __init__(self, connection: sqlite3.Connection, *, _legacy: bool = False) -> None:
        self._connection = connection
        self.units: dict[str, tuple[CorpusIngestUnit, ...]] = {}
        # Share immutable equal body text only; every revision and authority
        # container remains distinct. The pool is bounded by retained units.
        self._bodies: dict[str, str] = {}
        self._summaries: dict[str, dict] = {}
        self._records: dict[str, _ProgressRecord] = {}
        self._pair_roots: dict[str, _ProgressRecord] = {}
        self.portfolio: tuple[dict, ...] = ()
        self._portfolio_record: tuple[int, str] | None = None
        self.observations: dict[str, tuple[str, str, str, str]] = {}
        self._current_state = not _legacy
        if not _legacy:
            owns_read = not connection.in_transaction
            if owns_read:
                connection.execute('BEGIN')
            try:
                self._load_current()
            finally:
                if owns_read:
                    connection.rollback()
            _share_progress_strings(self._summaries)
            return
        for seq, kind, raw, payload_digest in connection.execute(
            "SELECT seq,kind,payload_json,payload_digest FROM ledger "
            "WHERE kind IN (?,?,?) ORDER BY seq", (LAND, STATE, PORTFOLIO),
        ):
            if digest_bytes(raw.encode()) != payload_digest:
                raise ValueError("native progress ledger payload differs")
            value = json.loads(raw)
            if canonical_json_bytes(value).decode() != raw:
                raise ValueError("native progress ledger is not canonical")
            self._apply(kind, value, seq=seq, payload_digest=payload_digest)
        _share_progress_strings(self._summaries)

    def _load_current(self) -> None:
        from .native_progress_state import checked_json, require_ready
        require_ready(self._connection)
        for revision, seq, digest, raw, content_digest in self._connection.execute(
            'SELECT revision_id,land_seq,land_digest,content_json,content_digest FROM native_current_sources ORDER BY land_seq'
        ):
            value = checked_json(raw, content_digest, label='source')
            self._apply(LAND, value, seq=seq, payload_digest=digest)
            if value['revision_id'] != revision:
                raise ValueError('native CURRENT source identity differs')
        expected_units = {unit.ingest_id: (unit.revision_id, digest_bytes(canonical_json_bytes(asdict(unit.effective_revision))))
                          for units in self.units.values() for unit in units}
        indexed_units = {ingest: (revision, digest) for ingest, revision, digest in self._connection.execute(
            'SELECT ingest_id,revision_id,effective_revision_digest FROM native_current_units')}
        if indexed_units != expected_units:
            raise ValueError('native CURRENT source index binding differs')
        for revision, ordinal, seq, digest, raw, state_digest, pair_digest in self._connection.execute(
            'SELECT revision_id,ordinal,diagnostic_seq,diagnostic_digest,state_json,state_digest,pair_digest FROM native_current_heads ORDER BY diagnostic_seq'
        ):
            value = json.loads(raw)
            if (type(value) is not dict or canonical_json_bytes(value).decode() != raw
                    or value.get('revision_id') != revision or type(value.get('ordinal')) is not int
                    or value['ordinal'] != ordinal or ordinal < 1 or type(seq) is not int or seq < 0
                    or type(value.get('facts')) is not dict or not value.get('stage')
                    or revision not in self.units
                    or any(key in value['facts'] for key in _RETRIEVAL_FIELDS) and pair_digest is not None
                    or digest_bytes(canonical_json_bytes({'state': value, 'pair_digest': pair_digest})) != state_digest):
                raise ValueError('native CURRENT head payload differs')
            self._records[revision] = _ProgressRecord(seq, digest, ordinal, pair_digest,
                _state_digest(value['stage'], value['facts'], pair_digest), None)
            self._summaries[revision] = value
        row = self._connection.execute('SELECT diagnostic_seq,diagnostic_digest,portfolio_json,portfolio_digest FROM native_current_portfolio WHERE singleton=1').fetchone()
        if row:
            seq, digest, raw, content_digest = row
            self._apply(PORTFOLIO, checked_json(raw, content_digest, label='portfolio'), seq=seq, payload_digest=digest)
        for observation, raw, digest in self._connection.execute('SELECT observation_digest,reference_json,reference_digest FROM native_current_observations'):
            value = checked_json(raw, digest, label='observation')['reference']
            if len(value) != 4 or any(type(item) is not str or not item for item in value) or value[1] != observation:
                raise ValueError('native CURRENT observation binding differs')
            self.observations[observation] = tuple(value)

    def _apply(self, kind: str, value: dict, *, seq: int, payload_digest: str) -> None:
        if kind == LAND:
            # Chunk receipts repeat the full source body. Share exact-equal text
            # across revisions too; retain and validate every original ledger byte.
            units = _landed_units(value, bodies=self._bodies)
            self._validate_units(units)
            revision_id = units[0].revision_id
            if value["revision_id"] != revision_id:
                raise ValueError("native progress revision identity differs")
            prior = self.units.get(revision_id)
            if prior is not None and prior != units:
                raise ValueError("native progress retained units changed")
            self.units[revision_id] = units
        elif kind == STATE:
            revision_id = value["revision_id"]
            if revision_id not in self.units:
                raise ValueError("native progress lacks its landed revision")
            previous = self._records.get(revision_id)
            ordinal = value["ordinal"]
            if type(ordinal) is not int or ordinal != (previous.ordinal if previous else 0) + 1:
                raise ValueError("native progress ordinal has a gap")
            if type(value.get("facts")) is not dict:
                raise ValueError("native progress facts must be an object")
            facts = dict(value["facts"])
            if "retrieval_facts_ref" in value:
                reference = value["retrieval_facts_ref"]
                if (
                    type(reference) is not dict
                    or previous is None or previous.pair_digest is None
                    or type(reference.get("seq")) is not int
                    or type(reference.get("ordinal")) is not int
                    or reference != previous.reference()
                    or reference["seq"] >= seq
                    or any(key in facts for key in _RETRIEVAL_FIELDS)
                ):
                    raise ValueError("native retrieval facts reference differs")
                if revision_id not in self._pair_roots:
                    raise ValueError("native retrieval facts reference lacks its root")
                pair_digest = previous.pair_digest
                pair_root = previous.pair_root
            else:
                pair_digest = _pair_digest(facts)
                pair_root = None if pair_digest is None else (seq, payload_digest, ordinal)
            logical = {key: item for key, item in value.items() if key != "retrieval_facts_ref"}
            logical["facts"] = facts
            record = _ProgressRecord(
                seq, payload_digest, ordinal, pair_digest,
                _state_digest(value["stage"], facts, pair_digest),
                pair_root,
            )
            self._records[revision_id] = record
            if pair_digest is None:
                self._pair_roots.pop(revision_id, None)
            elif "retrieval_facts_ref" not in value:
                self._pair_roots[revision_id] = record
            logical["facts"] = {
                key: item for key, item in facts.items()
                if pair_digest is None or key not in _RETRIEVAL_FIELDS
            }
            # An advance result and its input are ordinary mutable DTOs, not a
            # view of retained state. Only the small inline tree remains here.
            self._summaries[revision_id] = deepcopy(logical)
        elif kind == PORTFOLIO:
            self.portfolio = tuple(value["sources"])
            for source in self.portfolio:
                for raw in source.get("observations", ()):
                    observation = tuple(raw)
                    if len(observation) != 4 or any(type(item) is not str or not item for item in observation):
                        raise ValueError("native source observation reference differs")
                    # Retain the first exact observation, also after a page
                    # leaves the feed. This is a reference, not a second copy.
                    self.observations.setdefault(observation[1], observation)
            self._portfolio_record = (seq, payload_digest)

    def portfolio_reference(self, sources: tuple[dict, ...]) -> dict:
        if (
            self._portfolio_record is None or sources != self.portfolio
            or digest_bytes(canonical_json_bytes({"sources": list(sources)}))
            != self._portfolio_record[1]
        ):
            raise ValueError("native source portfolio reference differs")
        return {"seq": self._portfolio_record[0],
                "payload_digest": self._portfolio_record[1]}

    @staticmethod
    def _validate_units(units: tuple[CorpusIngestUnit, ...]) -> None:
        if not units or any(type(unit) is not CorpusIngestUnit or unit.authority is None for unit in units):
            raise ValueError("native progress needs exact governed units")
        first = units[0]
        if (
            tuple(unit.chunk_ordinal for unit in units) != tuple(range(1, first.chunk_count + 1))
            or any(unit.revision_id != first.revision_id or unit.chunk_count != first.chunk_count for unit in units)
            or any(not unit.proving_run_id.startswith("native-source:") for unit in units)
        ):
            raise ValueError("native progress revision chunk coverage differs")
        revision_digest = first.revision_digest
        if any(unit.revision_digest != revision_digest for unit in units[1:]):
            raise ValueError("native progress revision chunk coverage differs")

    def _retain(self, kind: str, value: dict) -> None:
        from .native_progress_state import (retain_source, retain_head, retain_portfolio,
            retain_embedding_pins, write_counts)
        diagnostic = value
        pair_digest = _pair_digest(value.get('facts', {})) if kind == STATE else None
        if kind == STATE:
            inline = {key: item for key, item in value['facts'].items()
                      if pair_digest is None or key not in _RETRIEVAL_FIELDS}
            diagnostic = {key: value[key] for key in ('revision_id', 'ordinal', 'stage')}
            diagnostic['state_digest'] = _state_digest(value['stage'], inline, pair_digest)
            diagnostic['retrieval_pair_digest'] = pair_digest
            if value['stage'] in {'EMBEDDING_STARTED', 'ASSESSMENT_STARTED', 'PUBLICATION_STARTED', 'ACKNOWLEDGED', 'COPY_CORRECTION_PREPARED'}:
                diagnostic['facts'] = inline
        try:
            if kind == STATE and 'facts' not in diagnostic:
                if not self._connection.in_transaction:
                    self._connection.execute('BEGIN IMMEDIATE')
                seq, digest = 0, digest_bytes(canonical_json_bytes(diagnostic))
            else:
                append_ledger(self._connection, kind, diagnostic)
                seq, digest = self._connection.execute('SELECT seq,payload_digest FROM ledger WHERE seq=last_insert_rowid()').fetchone()
            if kind == LAND:
                retain_source(self._connection, value, seq=seq, digest=digest, units=_landed_units(value))
            elif kind == STATE:
                retain_head(self._connection, value, seq=seq, digest=digest, pair_digest=pair_digest,
                            expected_previous_ordinal=value['ordinal'] - 1)
                retain_embedding_pins(self._connection, diagnostic, seq=seq, digest=digest)
            else:
                retain_portfolio(self._connection, value, seq=seq, digest=digest)
            write_counts(self._connection)
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        if kind == STATE and seq == 0:
            # Optional diagnostics happen after the business transaction. A
            # missing/failed log sink must never roll back committed state.
            try:
                from .diagnostic_logging import emit_diagnostic
                emit_diagnostic('native_revision_progress', diagnostic)
            except Exception:
                pass
        if kind != STATE:
            self._apply(kind, value, seq=seq, payload_digest=digest)
        else:
            self._records[value['revision_id']] = _ProgressRecord(seq, digest, value['ordinal'], pair_digest,
                _state_digest(value['stage'], value['facts'], pair_digest), None)
            self._summaries[value['revision_id']] = deepcopy({**value, 'facts': inline})

    def land(self, units: tuple[CorpusIngestUnit, ...]) -> None:
        units = tuple(sorted(units, key=lambda unit: unit.chunk_ordinal))
        self._validate_units(units)
        revision_id = units[0].revision_id
        prior = self.units.get(revision_id)
        if prior is not None:
            # Re-observation HTTP/access receipts may change; retained source
            # and representation identities may not silently change here.
            if tuple((unit.ingest_id, unit.authority.representation_id) for unit in prior) != tuple((unit.ingest_id, unit.authority.representation_id) for unit in units):
                raise ValueError("native progress revision was rebound")
            return
        value = {"revision_id": revision_id, "units": [asdict(unit) for unit in units]}
        if len(units) > 1:
            # A 43-chunk workbook otherwise writes its full body 43 times.
            # Keep all individual receipts and restore byte-identical logical
            # units on replay; no historical rows or ledger hashes are rewritten.
            if any(unit.body != units[0].body for unit in units):
                raise ValueError("native progress shared body differs")
            value["shared_body"] = units[0].body
            for item in value["units"]:
                del item["body"]
        self._retain(LAND, value)

    def progress_ordinal(self, revision_id: str) -> int | None:
        """Return retained progress identity, not mutable public logical facts."""
        record = self._records.get(revision_id)
        return None if record is None else record.ordinal

    def summary(self, revision_id: str) -> dict:
        """Return detached inline continuation metadata, never a full pair."""
        logical = self._summaries.get(revision_id)
        record = self._records.get(revision_id)
        if logical is None:
            if record is not None:
                raise ValueError("native progress current summary is missing")
            return {}
        if record is None:
            raise ValueError("native progress current record is missing")
        if (
            logical.get("revision_id") != revision_id
            or type(logical.get("ordinal")) is not int
            or logical["ordinal"] != record.ordinal
            or _state_digest(logical["stage"], logical["facts"], record.pair_digest)
            != record.state_digest
        ):
            raise ValueError("native progress current logical state differs")
        return deepcopy(logical)

    def iter_summaries(self) -> Iterator[tuple[str, dict]]:
        """Yield detached metadata in retained order over a snapshot of keys."""
        if self._summaries.keys() != self._records.keys():
            raise ValueError("native progress current metadata inventory differs")
        return ((revision_id, self.summary(revision_id))
                for revision_id in tuple(self._summaries))

    def _pair_facts(self, revision_id: str) -> dict:
        record = self._records[revision_id]
        if self._current_state:
            from .native_progress_state import checked_json
            row = self._connection.execute('SELECT pair_json FROM native_current_pairs WHERE pair_digest=?', (record.pair_digest,)).fetchone()
            if row is None:
                raise ValueError('native CURRENT retrieval pair root is missing')
            value = checked_json(row[0], record.pair_digest, label='retrieval pair')
            if set(value) != set(_RETRIEVAL_FIELDS):
                raise ValueError('native CURRENT retrieval pair fields differ')
            return value
        root = self._pair_roots.get(revision_id)
        if (
            type(root) is not _ProgressRecord
            or type(root.seq) is not int or type(root.ordinal) is not int
            or root.seq < 1 or root.ordinal < 1
            or root.seq > record.seq or root.ordinal > record.ordinal
            or (root.seq, root.payload_digest, root.ordinal) != record.pair_root
            or root.pair_digest != record.pair_digest
        ):
            raise ValueError("native retrieval pair root differs")
        row = self._connection.execute(
            "SELECT kind,payload_json,payload_digest FROM ledger WHERE seq=?",
            (root.seq,),
        ).fetchone()
        if (
            row is None or row[0] != STATE or row[2] != root.payload_digest
            or digest_bytes(row[1].encode()) != root.payload_digest
        ):
            raise ValueError("native retrieval pair root payload differs")
        value = json.loads(row[1])
        if (
            type(value) is not dict or canonical_json_bytes(value).decode() != row[1]
            or value.get("revision_id") != revision_id
            or type(value.get("ordinal")) is not int or value["ordinal"] != root.ordinal
            or type(value.get("facts")) is not dict
            or "retrieval_facts_ref" in value
            or _pair_digest(value["facts"]) != record.pair_digest
            or _state_digest(value["stage"], value["facts"], root.pair_digest)
            != root.state_digest
        ):
            raise ValueError("native retrieval pair root body differs")
        return {key: value["facts"][key] for key in _RETRIEVAL_FIELDS}

    def current(self, revision_id: str) -> dict:
        """Return complete detached facts through one selected verified root PK."""
        logical = self.summary(revision_id)
        if logical and self._records[revision_id].pair_digest is not None:
            logical["facts"].update(self._pair_facts(revision_id))
        return logical

    def advance(self, revision_id: str, *, stage: str, facts: dict) -> dict:
        if revision_id not in self.units or not stage:
            raise ValueError("native progress stage lacks a landed revision")
        if type(facts) is not dict:
            raise ValueError("native progress facts must be an object")
        self.summary(revision_id)
        previous = self._records.get(revision_id)
        facts = json.loads(canonical_json_bytes(facts))
        pair_digest = _pair_digest(facts)
        if previous and pair_digest is not None and pair_digest == previous.pair_digest:
            # Check the referenced immutable bytes before creating any new
            # reference or accepting a no-op, not after the append commits.
            self._pair_facts(revision_id)
        logical = {"revision_id": revision_id, "ordinal": previous.ordinal if previous else 0,
                   "stage": stage, "facts": facts}
        if previous and _state_digest(stage, facts, pair_digest) == previous.state_digest:
            return logical
        logical["ordinal"] += 1
        self._retain(STATE, logical)
        return logical

    def sources(self, dispositions: tuple) -> None:
        values = tuple({
            "source_id": item.source_id, "status": item.status,
            "reason_code": item.reason_code,
            "revision_ids": sorted({unit.revision_id for unit in item.units}),
            "observations": [list(value) for value in getattr(item, "observations", ())],
            "item_holds": [list(value) for value in getattr(item, "item_holds", ())],
        } for item in dispositions)
        if self._portfolio_record is None or values != self.portfolio:
            self._retain(PORTFOLIO, {"sources": list(values)})


def import_legacy_native_progress(connection: sqlite3.Connection) -> dict:
    """One explicit, checked, atomic conversion; ordinary boot never invokes it."""
    import time
    from .native_progress_state import (ensure_schema, require_ready, retain_source, retain_head,
        retain_portfolio, retain_embedding_pins, write_counts)
    started = time.monotonic()
    ensure_schema(connection)
    if connection.execute('SELECT 1 FROM native_current_meta WHERE singleton=1').fetchone():
        return {'already_current': True, 'counts': require_ready(connection), 'elapsed_seconds': time.monotonic()-started}
    from .native_progress_state import TABLES
    if any(connection.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone()
           for table in (*TABLES, 'native_current_units')):
        raise ValueError('native CURRENT marker missing from populated state; explicit recovery required')
    try:
        connection.execute('BEGIN IMMEDIATE')
        journal = NativeRevisionJournal(connection, _legacy=True)
        for seq, raw, digest in connection.execute('SELECT seq,payload_json,payload_digest FROM ledger WHERE kind=? ORDER BY seq', (LAND,)):
            value = json.loads(raw)
            if not connection.execute('SELECT 1 FROM native_current_sources WHERE revision_id=?', (value['revision_id'],)).fetchone():
                retain_source(connection, value, seq=seq, digest=digest, units=journal.units[value['revision_id']])
        for revision, record in journal._records.items():
            retain_head(connection, journal.current(revision), seq=record.seq, digest=record.payload_digest, pair_digest=record.pair_digest)
        for seq, raw, digest in connection.execute("SELECT seq,payload_json,payload_digest FROM ledger WHERE kind=? AND json_extract(payload_json,'$.stage')='EMBEDDING_STARTED' ORDER BY seq", (STATE,)):
            retain_embedding_pins(connection, json.loads(raw), seq=seq, digest=digest)
        if journal._portfolio_record is not None:
            seq, digest = journal._portfolio_record
            retain_portfolio(connection, {'sources': list(journal.portfolio)}, seq=seq, digest=digest)
        for reference in journal.observations.values():
            raw = canonical_json_bytes({'reference': list(reference)}).decode()
            connection.execute('INSERT OR IGNORE INTO native_current_observations VALUES(?,?,?)', (reference[1], raw, digest_bytes(raw.encode())))
        write_counts(connection)
        counts = require_ready(connection)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return {'already_current': False, 'counts': counts, 'elapsed_seconds': time.monotonic()-started}
