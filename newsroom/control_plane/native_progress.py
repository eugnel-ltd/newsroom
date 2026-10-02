"""Native revision continuation in the existing append-only Control Plane ledger.

Load once at daemon start; append only changed state. Source bytes are retained
once, never copied into each stage receipt. No additional database or schema.
Unchanged retrieval binding/rights pairs refer to the previous same-revision
progress record. Ordered replay verifies their original immutable root; selected
reads restore complete detached facts without retaining every cold pair in RAM.
The journal records work, not evidence/publication authority.
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
    def __init__(self, connection: sqlite3.Connection) -> None:
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
        # ponytail: one startup replay; an indexed snapshot is warranted only
        # after measured native history makes this bounded-kind scan material.
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
            else:
                pair_digest = _pair_digest(facts)
            logical = {key: item for key, item in value.items() if key != "retrieval_facts_ref"}
            logical["facts"] = facts
            record = _ProgressRecord(
                seq, payload_digest, ordinal, pair_digest,
                _state_digest(value["stage"], facts, pair_digest),
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
        # Existing store writer/chain semantics own the atomic append. Apply
        # only after commit; an interrupted commit is reconstructed on reopen.
        try:
            append_ledger(self._connection, kind, value)
            seq, payload_digest = self._connection.execute(
                "SELECT seq,payload_digest FROM ledger WHERE seq=last_insert_rowid()"
            ).fetchone()
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        self._apply(kind, value, seq=seq, payload_digest=payload_digest)

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
        root = self._pair_roots.get(revision_id)
        if (
            type(root) is not _ProgressRecord
            or type(root.seq) is not int or type(root.ordinal) is not int
            or root.seq < 1 or root.ordinal < 1
            or root.seq > record.seq or root.ordinal > record.ordinal
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
        encoded = logical
        if (
            previous and pair_digest is not None and pair_digest == previous.pair_digest
        ):
            encoded = {**logical,
                       "facts": {key: item for key, item in facts.items() if key not in _RETRIEVAL_FIELDS},
                       "retrieval_facts_ref": previous.reference()}
        self._retain(STATE, encoded)
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
