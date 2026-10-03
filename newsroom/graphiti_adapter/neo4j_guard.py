"""Durable Neo4j mutation journal and exact Graphiti completion marker."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4
from time import perf_counter_ns, process_time_ns

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes


_RESERVED_PREFIX = "_newsroom_"
_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SNAPSHOT_NODE = "NewsroomSnapshotNode"
_SNAPSHOT_RELATIONSHIP = "NewsroomSnapshotRelationship"
_MARKER = "NewsroomIngestMarker"
_MARKER_CLAIM_LEASE = "PT15M"
_LOGGER = logging.getLogger("newsroom.diagnostic")
# ponytail: fixed conservative write bounds; tune only against retained service evidence.
_PAGE_TARGET_LIMIT = 64
_PAGE_PROPERTY_BYTES = 8 * 1024 * 1024
_INVENTORY_BYTES = 8 * 1024 * 1024
_UNRESOLVED_STATES = ('SNAPSHOTTING', 'PENDING', 'ROLLING_BACK', 'RECOVERING')
_OWNED_RECOVERY_PHASES = frozenset({
    "OWNERSHIP", "INVENTORY", "DELETE_NEW_RELATIONSHIPS", "DELETE_NEW_NODES",
    "RESTORE", "FULL_VERIFY", "TERMINAL", "SNAPSHOT_CLEANUP",
})
_SCHEMA_QUERIES = (
    f"""
    CREATE CONSTRAINT newsroom_ingest_marker_episode IF NOT EXISTS
    FOR (m:{_MARKER}) REQUIRE m.episode_uuid IS UNIQUE
    """,
    f"""
    CREATE INDEX newsroom_snapshot_node_identity IF NOT EXISTS
    FOR (s:{_SNAPSHOT_NODE}) ON (s._newsroom_snapshot_id)
    """,
    f"""
    CREATE INDEX newsroom_snapshot_relationship_identity IF NOT EXISTS
    FOR (s:{_SNAPSHOT_RELATIONSHIP}) ON (s._newsroom_snapshot_id)
    """,
)


@contextmanager
def _guard_phase(phase: str, *, episode_id: str, attempt_number: int):
    """Optional inclusive timing only; never mutation or recovery authority."""
    started = None
    try:
        started = (perf_counter_ns(), process_time_ns())
    except Exception:
        pass
    status, failure = "FAILED", "NONE"
    try:
        yield
        status = "COMPLETE"
    except BaseException as exc:
        failure = type(exc).__name__
        raise
    finally:
        if started is not None:
            try:
                data = {
                    "phase": phase, "episode_id": episode_id[:128],
                    "attempt_number": attempt_number, "status": status,
                    "failure_class": failure,
                    "elapsed_ms": (perf_counter_ns() - started[0]) // 1_000_000,
                    "cpu_ms": (process_time_ns() - started[1]) // 1_000_000,
                    "cpu_scope": "PROCESS", "nested_spans_not_additive": True,
                }
                _LOGGER.info("graphiti_guard_phase", extra={
                    "diagnostic_event": "graphiti_guard_phase", "diagnostic_data": data,
                })
            except Exception:
                pass


class GuardError(RuntimeError):
    """The proposal generation could not be proved unchanged or recoverable."""


_OWNED_GUARD_REASON_CODES = {
    "Graphiti identity inventory exceeds its byte bound": "INVENTORY_BYTE_BOUND",
    "Graphiti snapshot coverage count is invalid": "SNAPSHOT_COVERAGE",
    "Graphiti snapshot coverage identity is absent": "SNAPSHOT_COVERAGE",
    "Graphiti identity inventory omits a pre-existing target": "SNAPSHOT_COVERAGE",
    "Graphiti target exceeds the full property byte bound": "PROPERTY_BYTE_BOUND",
    "a pre-existing Graphiti node is missing": "TARGET_MISSING",
    "a pre-existing Graphiti relationship is missing": "TARGET_MISSING",
    "Graphiti bounded property write lost an actual target": "TARGET_MISSING",
    "Graphiti property read lost an inventoried actual pair": "TARGET_MISSING",
    "Graphiti label repair lost an actual target": "TARGET_MISSING",
    "a pre-existing Graphiti node changed across the attempt": "TARGET_DRIFT",
    "a pre-existing Graphiti relationship changed across the attempt": "TARGET_DRIFT",
    "Graphiti inventory identity is absent": "IDENTITY_BINDING",
    "Graphiti inventory has a duplicate actual pair": "IDENTITY_BINDING",
    "Graphiti guard marker identity is absent": "IDENTITY_BINDING",
    "Graphiti owned compensation identity is malformed": "IDENTITY_BINDING",
    "Graphiti owned compensation snapshot is malformed": "IDENTITY_BINDING",
    "Graphiti guard snapshot identity is malformed": "IDENTITY_BINDING",
    "Graphiti guard marker identity differs from this input": "IDENTITY_BINDING",
    "Graphiti attempt marker identity differs": "IDENTITY_BINDING",
    "Graphiti guard marker is malformed": "IDENTITY_BINDING",
}


def _owned_recovery_guard_reason(error: GuardError) -> str:
    message = error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else None
    return _OWNED_GUARD_REASON_CODES.get(message, "GUARD_ERROR")


class GuardState(StrEnum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    ROLLING_BACK = "ROLLING_BACK"
    COMPLETE = "COMPLETE"
    RECOVERED_AMBIGUOUS = "RECOVERED_AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class GuardMarker:
    state: GuardState
    attempt_number: int
    input_digest: str
    chat_invocations: tuple[dict[str, object], ...] = ()
    embedding_usage: dict[str, object] | None = None


def _normalise(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return value


def _property_bytes(value: object) -> int:
    """Conservative full-value bound, including containers and overwritten values."""
    if isinstance(value, dict):
        return 64 + sum(64 + len(str(key).encode("utf-8")) + _property_bytes(item)
                        for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return 64 + sum(16 + _property_bytes(item) for item in value)
    if isinstance(value, (bytes, bytearray, str)):
        return 64 + len(value.encode("utf-8") if isinstance(value, str) else value)
    if value is None or isinstance(value, (bool, int, float)):
        return 16
    # Native vectors expose their full raw bytes; their display may truncate.
    # Use the value protocol without importing the optional graph runtime here.
    raw = getattr(value, "raw", None)
    if callable(raw):
        buffer = raw()
        if isinstance(buffer, (bytes, bytearray, memoryview)):
            return 64 + memoryview(buffer).nbytes
    # Neo4j temporal/spatial scalars are retained server-side, not serialised back.
    return 64 + len(str(value).encode("utf-8"))


def _record_value(record: object, key: str) -> object:
    if isinstance(record, dict):
        return record.get(key)
    try:
        return record[key]  # type: ignore[index]
    except (KeyError, TypeError):
        return None


class Neo4jMutationGuard:
    """Journal a generation before provider work and restore existing values."""

    __slots__ = (
        "_attempt_number",
        "_claim_token",
        "_driver",
        "_episode_uuid",
        "_fence_transaction",
        "_generation_key",
        "_snapshot_cleanup_pending",
        "_group_id",
        "_input_digest",
        "_marker_episode_uuid",
        "_owned_recovery_stop_check",
        "_owned_recovery_phase_observer",
        "_snapshot_id",
    )

    def __init__(
        self,
        driver: Any,
        *,
        group_id: str,
        episode_uuid: str,
        attempt_number: int,
        input_digest: str,
        marker_episode_uuid: str | None = None,
    ) -> None:
        self._driver = driver
        self._claim_token: str | None = None
        self._fence_transaction: Any | None = None
        self._snapshot_cleanup_pending = False
        self._owned_recovery_stop_check: Callable[[], None] | None = None
        self._owned_recovery_phase_observer: Callable[[str], None] | None = None
        self._generation_key = "generation-owner:v1:" + digest_bytes(group_id.encode("utf-8"))
        self._group_id = group_id
        self._episode_uuid = episode_uuid
        self._marker_episode_uuid = marker_episode_uuid or episode_uuid
        if not self._marker_episode_uuid:
            raise GuardError("Graphiti guard marker identity is absent")
        self._attempt_number = attempt_number
        self._input_digest = input_digest
        self._snapshot_id = f"{episode_uuid}:{attempt_number}"

    @property
    def driver(self) -> Any:
        return self._driver

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def episode_uuid(self) -> str:
        return self._episode_uuid

    @property
    def input_digest(self) -> str:
        return self._input_digest

    async def _query(self, query: str, **parameters: object) -> list[object]:
        if self._owned_recovery_stop_check is not None:
            self._owned_recovery_stop_check()
        records, _, _ = await self._driver.execute_query(
            query, params=parameters, routing_="w"
        )
        return list(records)

    async def _stream_query(
        self, query: str, validate: Callable[[object], None],
        *, snapshot_label: str,
        **parameters: object,
    ) -> None:
        async def consume(transaction: Any) -> None:
            if self._owned_recovery_stop_check is not None:
                self._owned_recovery_stop_check()
            # Reinitialise coverage on every managed transaction attempt. MATCH
            # avoids correlated full scans, but omitted originals must still fail.
            count_result = await transaction.run(
                f"""
                MATCH (s:{snapshot_label} {{_newsroom_snapshot_id: $snapshot_id}})
                RETURN count(s) AS snapshot_count
                """,
                **parameters,
            )
            count_record = await count_result.single(strict=True)
            expected_count = _record_value(count_record, "snapshot_count")
            if type(expected_count) is not int or expected_count < 0:
                raise GuardError("Graphiti snapshot coverage count is invalid")
            covered: set[str] = set()
            records = await transaction.run(query, **parameters)
            async for record in records:
                if self._owned_recovery_stop_check is not None and len(covered) % _PAGE_TARGET_LIMIT == 0:
                    self._owned_recovery_stop_check()
                validate(record)
                identity = _record_value(record, "snapshot_identity")
                if not isinstance(identity, str) or not identity:
                    raise GuardError("Graphiti snapshot coverage identity is absent")
                covered.add(identity)
            if len(covered) != expected_count:
                kind = "node" if snapshot_label == _SNAPSHOT_NODE else "relationship"
                raise GuardError(f"a pre-existing Graphiti {kind} is missing")

        async with self._driver.session() as session:
            await session.execute_write(consume)

    def _require_pending_claim(self, records: list[object], *, operation: str) -> None:
        if (
            not records
            or _record_value(records[0], "claim_token") != self._claim_token
        ):
            raise GuardError(f"Graphiti {operation} lost its pending claim")

    @staticmethod
    async def bootstrap_schema(driver: Any) -> None:
        """Create journal schema once during explicit Neo4j bootstrap."""

        for query in _SCHEMA_QUERIES:
            await driver.execute_query(query, params={}, routing_="w")

    async def _marker(self) -> dict[str, object] | None:
        records = await self._query(
            f"""
            MATCH (m:{_MARKER} {{episode_uuid: $episode_uuid}})
            RETURN properties(m) AS marker
            """,
            episode_uuid=self._marker_episode_uuid,
        )
        if not records:
            return None
        marker = _record_value(records[0], "marker")
        return dict(marker) if isinstance(marker, dict) else None

    async def marker_exists(self) -> bool:
        """Report whether this exact internal attempt marker is retained."""

        return await self._marker() is not None

    async def recovered_ambiguous_marker_or_none(self) -> GuardMarker | None:
        """Read and validate an exactly retained completed rollback marker."""

        raw = await self._marker()
        if raw is None:
            return None
        marker = self._bind_marker(raw)
        return (
            marker
            if marker.state is GuardState.RECOVERED_AMBIGUOUS
            else None
        )

    @staticmethod
    async def owned_pending_identity(
        driver: Any, *, group_id: str,
    ) -> tuple[str, str, int] | None:
        """Read an expired exact durable owner, never create or claim a marker."""
        records, _, _ = await driver.execute_query(
            f"""
            MATCH (g:{_MARKER} {{episode_uuid: $generation_key}})
            MATCH (m:{_MARKER} {{episode_uuid: g.owner_marker_uuid}})
            WHERE g.group_id = $group_id AND m.group_id = $group_id
              AND g.snapshot_id = m.snapshot_id AND g.claim_token = m.claim_token
              AND m.state IN ['PENDING', 'ROLLING_BACK']
              AND (m.claim_expires_at IS NULL OR m.claim_expires_at <= datetime())
            RETURN m.episode_uuid AS marker_episode_uuid,
                   m.snapshot_id AS snapshot_id, m.attempt_number AS attempt_number
            """,
            params={"generation_key": "generation-owner:v1:" + digest_bytes(group_id.encode()),
                    "group_id": group_id}, routing_="w",
        )
        if not records:
            return None
        marker_uuid = _record_value(records[0], "marker_episode_uuid")
        snapshot = _record_value(records[0], "snapshot_id")
        attempt = _record_value(records[0], "attempt_number")
        if (not isinstance(marker_uuid, str) or not marker_uuid
            or not isinstance(snapshot, str) or type(attempt) is not int or attempt < 1):
            raise GuardError("Graphiti owned compensation identity is malformed")
        episode, separator, ordinal = snapshot.rpartition(":")
        if not separator or not episode or ordinal != str(attempt):
            raise GuardError("Graphiti owned compensation snapshot is malformed")
        return episode, marker_uuid, attempt

    async def recover_owned_pending(
        self, *, owner_stop_check: Callable[[], None] | None = None,
        phase_observer: Callable[[str], None] | None = None,
    ) -> GuardMarker | None:
        """Compensate this exact expired owner without a fresh snapshot or leaf."""
        self._owned_recovery_phase_observer = phase_observer
        try:
            self._observe_owned_recovery_phase("OWNERSHIP")
            if owner_stop_check is not None:
                owner_stop_check()
            identity = await self.owned_pending_identity(self._driver, group_id=self._group_id)
            if identity != (self._episode_uuid, self._marker_episode_uuid, self._attempt_number):
                return None
            raw = await self._marker()
            if raw is None or str(raw.get("state")) not in {"PENDING", "ROLLING_BACK"}:
                return None
            marker = self._bind_marker(raw)
            self._owned_recovery_stop_check = owner_stop_check
            taken_over = await self._take_over(raw, state=marker.state.value)
            if taken_over is None:
                return None
            await self.rollback_pending(
                chat_invocations=[dict(item) for item in marker.chat_invocations],
                embedding_usage=dict(marker.embedding_usage or {}),
                reason="EXACT_OWNED_COMPENSATION",
            )
            self._observe_owned_recovery_phase("TERMINAL")
            return await self.recovered_ambiguous_marker_or_none()
        finally:
            self._owned_recovery_stop_check = None
            self._owned_recovery_phase_observer = None

    def _observe_owned_recovery_phase(self, phase: str) -> None:
        if self._owned_recovery_phase_observer is not None and phase in _OWNED_RECOVERY_PHASES:
            self._owned_recovery_phase_observer(phase)

    def _generation_lock(self) -> str:
        # All coordinated writers acquire generation before episode. The WHERE
        # follows the dependent SET, so ownership is read after the write lock.
        return f"""
            MERGE (g:{_MARKER} {{episode_uuid: $generation_key}})
            ON CREATE SET g.group_id = $group_id
            SET g.lock_tick = coalesce(g.lock_tick, 0) + 1
            WITH g WHERE g.group_id = $group_id
        """

    def _owned_match(self, states: tuple[str, ...]) -> str:
        return self._generation_lock() + f"""
            MATCH (m:{_MARKER} {{episode_uuid: $episode_uuid}})
            WHERE g.owner_marker_uuid = $episode_uuid
              AND g.snapshot_id = $snapshot_id AND g.claim_token = $claim_token
              AND m.group_id = $group_id AND m.input_digest = $input_digest
              AND m.snapshot_id = $snapshot_id AND m.claim_token = $claim_token
              AND m.state IN {list(states)!r}
        """

    def _ownership_parameters(self) -> dict[str, object]:
        return dict(generation_key=self._generation_key, group_id=self._group_id,
                    episode_uuid=self._marker_episode_uuid, snapshot_id=self._snapshot_id,
                    input_digest=self._input_digest, claim_token=self._claim_token)

    async def _owned_query(self, query: str, **parameters: object) -> list[object]:
        if self._owned_recovery_stop_check is not None:
            self._owned_recovery_stop_check()
        parameters = self._ownership_parameters() | parameters
        if self._fence_transaction is None:
            return await self._query(query, **parameters)
        result = await self._fence_transaction.run(query, **parameters)
        record = await result.single()
        return [] if record is None else [record]

    async def _claim_marker(self) -> tuple[dict[str, object], bool, bool]:
        claim_token = str(uuid4())
        records = await self._query(
            self._generation_lock() + f"""
            OPTIONAL MATCH (retained:{_MARKER} {{episode_uuid: $episode_uuid}})
            WITH g, retained
            WHERE (
                g.owner_marker_uuid IS NULL AND retained IS NULL
                AND NOT EXISTS {{
                    MATCH (unresolved:{_MARKER})
                    WHERE unresolved.state IN {list(_UNRESOLVED_STATES)!r}
                      AND unresolved.snapshot_id = $snapshot_id
                }}
            ) OR (
                g.owner_marker_uuid = $episode_uuid
                AND g.snapshot_id = retained.snapshot_id
                AND g.claim_token = retained.claim_token
            )
            MERGE (m:{_MARKER} {{episode_uuid: $episode_uuid}})
            ON CREATE SET
                m.group_id = $group_id, m.attempt_number = $attempt_number,
                m.input_digest = $input_digest, m.snapshot_id = $snapshot_id,
                m.state = 'SNAPSHOTTING', m.chat_invocations_json = '[]',
                m.embedding_usage_json = 'null', m.claim_token = $claim_token,
                m.claim_expires_at = datetime() + duration($claim_lease)
            FOREACH (_ IN CASE WHEN m.claim_token = $claim_token THEN [1] ELSE [] END |
                SET g.owner_marker_uuid = $episode_uuid, g.snapshot_id = m.snapshot_id,
                    g.claim_token = m.claim_token)
            RETURN properties(m) AS marker, m.claim_token = $claim_token AS claimed,
                   m.claim_token <> $claim_token
                       AND m.claim_expires_at > datetime() AS active
            """,
            **(self._ownership_parameters() | dict(
                attempt_number=self._attempt_number, claim_token=claim_token,
                claim_lease=_MARKER_CLAIM_LEASE)),
        )
        if not records:
            raise GuardError("Graphiti generation is owned or has an unresolved legacy marker")
        marker = _record_value(records[0], "marker")
        if not isinstance(marker, dict):
            raise GuardError("Graphiti guard marker is malformed")
        claimed = _record_value(records[0], "claimed") is True
        if claimed:
            self._claim_token = claim_token
        return dict(marker), claimed, _record_value(records[0], "active") is True

    async def _take_over(
        self, raw: dict[str, object], *, state: str, require_expired: bool = True,
    ) -> dict[str, object] | None:
        claim_token = str(uuid4())
        records = await self._query(
            self._generation_lock() + f"""
            MATCH (m:{_MARKER} {{episode_uuid: $episode_uuid}})
            WHERE m.group_id = $group_id AND m.input_digest = $input_digest
              AND m.state = $retained_state AND m.snapshot_id = $snapshot_id
              AND coalesce(m.claim_token, '') = $retained_claim_token
              AND (
                  (g.owner_marker_uuid = $episode_uuid
                   AND g.snapshot_id = m.snapshot_id AND g.claim_token = m.claim_token)
                  OR (NOT $require_expired AND m.state = 'RECOVERED_AMBIGUOUS'
                      AND g.owner_marker_uuid IS NULL AND NOT EXISTS {{
                          MATCH (unresolved:{_MARKER})
                          WHERE unresolved.state IN {list(_UNRESOLVED_STATES)!r}
                            AND unresolved.snapshot_id = $snapshot_id
                      }})
              )
              AND (NOT $require_expired OR m.claim_expires_at IS NULL
                   OR m.claim_expires_at <= datetime())
            SET m.state = $state, m.claim_token = $claim_token,
                m.claim_expires_at = datetime() + duration($claim_lease),
                g.owner_marker_uuid = $episode_uuid, g.snapshot_id = m.snapshot_id,
                g.claim_token = $claim_token
            RETURN properties(m) AS marker
            """,
            **(self._ownership_parameters() | dict(
                retained_state=str(raw.get("state") or ""),
                snapshot_id=str(raw.get("snapshot_id") or ""),
                retained_claim_token=str(raw.get("claim_token") or ""), state=state,
                require_expired=require_expired, claim_token=claim_token,
                claim_lease=_MARKER_CLAIM_LEASE)),
        )
        if not records:
            return None
        marker = _record_value(records[0], "marker")
        if not isinstance(marker, dict):
            raise GuardError("Graphiti guard marker is malformed")
        self._claim_token = claim_token
        return dict(marker)

    async def _discard_taken_over_marker(self) -> None:
        records = await self._owned_query(
            self._owned_match(("RECOVERING",)) + """
            WITH g, m, m.episode_uuid AS episode_uuid
            REMOVE g.owner_marker_uuid, g.snapshot_id, g.claim_token
            DELETE m RETURN episode_uuid
            """,
        )
        if not records:
            raise GuardError("Graphiti recovery marker deletion did not commit")
        self._claim_token = None

    def _adopt_retained_snapshot(
        self, raw: dict[str, object], *, attempt_number: int
    ) -> None:
        snapshot_id = str(raw.get("snapshot_id") or "")
        expected = f"{self._episode_uuid}:{attempt_number}"
        if snapshot_id != expected:
            raise GuardError("Graphiti guard snapshot identity is malformed")
        self._snapshot_id = snapshot_id

    def _bind_marker(self, raw: dict[str, object]) -> GuardMarker:
        if (
            str(raw.get("group_id") or "") != self._group_id
            or str(raw.get("input_digest") or "") != self._input_digest
        ):
            raise GuardError("Graphiti guard marker identity differs from this input")
        try:
            state = GuardState(str(raw["state"]))
            attempt_number = int(raw["attempt_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GuardError("Graphiti guard marker is malformed") from exc
        if (
            self._marker_episode_uuid != self._episode_uuid
            and attempt_number != self._attempt_number
        ):
            raise GuardError("Graphiti attempt marker identity differs")
        self._adopt_retained_snapshot(raw, attempt_number=attempt_number)
        invocations: tuple[dict[str, object], ...] = ()
        embedding_usage: dict[str, object] | None = None
        try:
            parsed_invocations = json.loads(str(raw.get("chat_invocations_json") or "[]"))
            parsed_usage = json.loads(str(raw.get("embedding_usage_json") or "null"))
        except json.JSONDecodeError as exc:
            raise GuardError("Graphiti guard telemetry is malformed") from exc
        if isinstance(parsed_invocations, list) and all(
            isinstance(item, dict) for item in parsed_invocations
        ):
            invocations = tuple(dict(item) for item in parsed_invocations)
        if isinstance(parsed_usage, dict):
            embedding_usage = dict(parsed_usage)
        return GuardMarker(
            state=state,
            attempt_number=attempt_number,
            input_digest=self._input_digest,
            chat_invocations=invocations,
            embedding_usage=embedding_usage,
        )

    async def begin(self) -> GuardMarker:
        with _guard_phase("BEGIN", episode_id=self._episode_uuid, attempt_number=self._attempt_number):
            self._snapshot_id = f"{self._episode_uuid}:{self._attempt_number}"
            # Completed readers remain usable even while another episode owns the group.
            terminal = await self._marker()
            if terminal is not None and str(terminal.get("state")) in {
                "COMPLETE", "RECOVERED_AMBIGUOUS",
            }:
                marker = self._bind_marker(terminal)
                if marker.state is GuardState.COMPLETE or self._attempt_number <= marker.attempt_number:
                    await self._delete_snapshot()
                    return marker
                taken_over = await self._take_over(terminal, state="RECOVERING", require_expired=False)
                if taken_over is None:
                    raise GuardError("Graphiti generation cannot claim a new attempt")
                async with self._generation_fence(("RECOVERING",)):
                    await self._delete_snapshot()
                    await self._discard_taken_over_marker()
                return await self.begin()

            retained, claimed, active = await self._claim_marker()
            if not claimed:
                if active:
                    raise GuardError("Graphiti guard marker is owned by an active attempt")
                retained_state = str(retained.get("state"))
                if retained_state in {"SNAPSHOTTING", "RECOVERING"}:
                    if (str(retained.get("group_id") or "") != self._group_id
                        or str(retained.get("input_digest") or "") != self._input_digest):
                        raise GuardError("Graphiti guard marker identity differs from this input")
                    try:
                        retained_attempt = int(retained["attempt_number"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise GuardError("Graphiti guard marker is malformed") from exc
                    if self._marker_episode_uuid != self._episode_uuid and retained_attempt != self._attempt_number:
                        raise GuardError("Graphiti attempt marker identity differs")
                    self._adopt_retained_snapshot(retained, attempt_number=retained_attempt)
                    taken_over = await self._take_over(retained, state="RECOVERING")
                    if taken_over is None:
                        raise GuardError("Graphiti generation takeover lost its claim")
                    async with self._generation_fence(("RECOVERING",)):
                        await self._delete_snapshot()
                        await self._discard_taken_over_marker()
                    return await self.begin()
                marker = self._bind_marker(retained)
                if marker.state not in {GuardState.PENDING, GuardState.ROLLING_BACK}:
                    raise GuardError("Graphiti generation marker is not recoverable")
                taken_over = await self._take_over(retained, state=retained_state)
                if taken_over is None:
                    raise GuardError("Graphiti generation takeover lost its claim")
                return self._bind_marker(taken_over)

            async with self._generation_fence(("SNAPSHOTTING",)):
                await self._snapshot()
                pending = await self._owned_query(
                    self._owned_match(("SNAPSHOTTING",)) + """
                SET m.state = 'PENDING' RETURN m.state AS state
                """,
                )
                if not pending or _record_value(pending[0], "state") != "PENDING":
                    raise GuardError("Graphiti guard marker lost its claim before dispatch")
            return GuardMarker(state=GuardState.CREATED, attempt_number=self._attempt_number,
                               input_digest=self._input_digest)

    async def _snapshot(self) -> None:
        unsafe = await self._query(
            f"""
            MATCH (n)
            WHERE n.group_id = $group_id
              AND NOT n:{_SNAPSHOT_NODE}
              AND NOT n:{_SNAPSHOT_RELATIONSHIP}
              AND NOT n:{_MARKER}
              AND (
                  n.uuid IS NULL
                  OR any(key IN keys(n) WHERE key STARTS WITH $reserved_prefix)
              )
            RETURN count(n) AS unsafe_nodes
            """,
            group_id=self._group_id,
            reserved_prefix=_RESERVED_PREFIX,
        )
        if unsafe and int(_record_value(unsafe[0], "unsafe_nodes") or 0):
            raise GuardError(
                "Graphiti generation has no stable UUID or uses reserved guard properties"
            )
        unsafe_relationships = await self._query(
            """
            MATCH (a)-[r]->(b)
            WHERE (a.group_id = $group_id OR b.group_id = $group_id)
              AND (r.uuid IS NULL OR a.uuid IS NULL OR b.uuid IS NULL
                   OR any(key IN keys(r) WHERE key STARTS WITH $reserved_prefix))
            RETURN count(r) AS unsafe_relationships
            """,
            group_id=self._group_id,
            reserved_prefix=_RESERVED_PREFIX,
        )
        if unsafe_relationships and int(
            _record_value(unsafe_relationships[0], "unsafe_relationships") or 0
        ):
            raise GuardError("Graphiti generation relationship has no stable UUID or uses reserved guard properties")
        nodes = await self._capture_inventory(
            f"""
            MATCH (n) WHERE n.group_id = $group_id
              AND NOT n:{_SNAPSHOT_NODE} AND NOT n:{_SNAPSHOT_RELATIONSHIP} AND NOT n:{_MARKER}
            """, element="n",
        )
        async for records in self._inventory_pages(nodes,
            """
            UNWIND $identities AS row
            MATCH (n) WHERE elementId(n) = row.source_identity
            RETURN elementId(n) AS source_identity, elementId(n) AS target_identity,
                   properties(n) AS source_properties, labels(n) AS expected
            ORDER BY source_identity, target_identity
            """,
        ):
            await self._write_page(
                f"""
            UNWIND $page AS row
            MATCH (n) WHERE elementId(n) = row.source_identity
            CREATE (s:{_SNAPSHOT_NODE}) SET s = properties(n)
            SET s._newsroom_snapshot_id = $snapshot_id,
                s._newsroom_source_uuid = n.uuid, s._newsroom_source_labels = labels(n)
            RETURN count(s) AS written
                """, records,
            )
        nodes.clear()
        relationships = await self._capture_inventory(
            f"""
            MATCH (a)-[r]->(b)
            WHERE (a.group_id = $group_id OR b.group_id = $group_id)
              AND NOT a:{_SNAPSHOT_NODE} AND NOT b:{_SNAPSHOT_NODE} AND r.uuid IS NOT NULL
            """, element="r",
        )
        async for records in self._inventory_pages(relationships,
            """
            UNWIND $identities AS row
            MATCH (a)-[r]->(b) WHERE elementId(r) = row.source_identity
            RETURN elementId(r) AS source_identity, elementId(r) AS target_identity, properties(r) AS source_properties
            ORDER BY source_identity, target_identity
            """,
        ):
            await self._write_page(
                f"""
            UNWIND $page AS row
            MATCH (a)-[r]->(b) WHERE elementId(r) = row.source_identity
            CREATE (s:{_SNAPSHOT_RELATIONSHIP}) SET s = properties(r)
            SET s._newsroom_snapshot_id = $snapshot_id, s._newsroom_relationship_uuid = r.uuid,
                s._newsroom_source_uuid = a.uuid, s._newsroom_target_uuid = b.uuid,
                s._newsroom_relationship_type = type(r)
            RETURN count(s) AS written
                """, records,
            )

    async def _capture_inventory(self, match: str, *, element: str) -> list[tuple[str, str]]:
        """Select the unchanged protected set once, without transferring properties."""
        identity = f"elementId({element})"
        async def consume(transaction):
            if self._owned_recovery_stop_check is not None:
                self._owned_recovery_stop_check()
            result = await transaction.run(match + " RETURN count(*) AS snapshot_count", group_id=self._group_id)
            count = _record_value(await result.single(strict=True), "snapshot_count")
            if type(count) is not int or count < 0:
                raise GuardError("Graphiti snapshot coverage count is invalid")
            records = await transaction.run(match + f" RETURN {identity} AS source_identity", group_id=self._group_id)
            identities, seen, size = [], set(), 256
            async for record in records:
                if self._owned_recovery_stop_check is not None and len(identities) % _PAGE_TARGET_LIMIT == 0:
                    self._owned_recovery_stop_check()
                source = _record_value(record, "source_identity")
                if type(source) is not str or not source:
                    raise GuardError("Graphiti inventory identity is absent")
                if source in seen:
                    raise GuardError("Graphiti inventory has a duplicate actual pair")
                size += 512 + 3 * len(source.encode())
                if size > _INVENTORY_BYTES:
                    raise GuardError("Graphiti identity inventory exceeds its byte bound")
                identities.append((source, source))
                seen.add(source)
            if len(identities) != count:
                raise GuardError("Graphiti identity inventory omits a pre-existing target")
            identities.sort()
            return identities
        async with self._driver.session() as session:
            return await session.execute_write(consume)

    async def record_pending_telemetry(
        self,
        *,
        chat_invocations: list[dict[str, object]],
        embedding_usage: dict[str, object],
    ) -> None:
        recorded = await self._owned_query(
            self._owned_match(("PENDING",)) + """
            SET m.chat_invocations_json = $chat_invocations_json,
                m.embedding_usage_json = $embedding_usage_json
            RETURN m.claim_token AS claim_token
            """,
            chat_invocations_json=canonical_json_bytes(chat_invocations).decode("utf-8"),
            embedding_usage_json=canonical_json_bytes(embedding_usage).decode("utf-8"),
        )
        self._require_pending_claim(recorded, operation="telemetry")

    @asynccontextmanager
    async def _generation_fence(self, states: tuple[str, ...]) -> AsyncIterator[None]:
        if self._fence_transaction is not None:
            yield
            return
        async with self._driver.session() as session:
            transaction = await session.begin_transaction()
            try:
                result = await transaction.run(
                    self._owned_match(states) + " RETURN m.claim_token AS claim_token",
                    **self._ownership_parameters(),
                )
                record = await result.single()
                self._require_pending_claim([] if record is None else [record], operation="generation mutation")
                self._fence_transaction = transaction
                yield
                await transaction.commit()
            except BaseException:
                self._snapshot_cleanup_pending = False
                await transaction.rollback()
                raise
            finally:
                self._fence_transaction = None
        if self._snapshot_cleanup_pending:
            self._snapshot_cleanup_pending = False
            self._observe_owned_recovery_phase("SNAPSHOT_CLEANUP")
            await self._delete_snapshot()

    @asynccontextmanager
    async def fenced_graph_mutation(self) -> AsyncIterator[None]:
        """Hold only the generation write lock; property pages commit independently."""
        async with self._generation_fence(("PENDING", "ROLLING_BACK")):
            yield

    async def restore_preexisting(self) -> None:
        """Restore every pre-attempt node/edge property while retaining new objects."""

        async with self.fenced_graph_mutation():
            await self._restore_properties()
            await self.assert_preexisting_unchanged()

    async def rollback_pending(
        self,
        *,
        chat_invocations: list[dict[str, object]],
        embedding_usage: dict[str, object],
        reason: str,
    ) -> bool:
        """Restore the exact pre-attempt generation and retain a recovery marker."""

        if self._fence_transaction is not None:
            raise GuardError("Graphiti rollback must enter after the graph mutation fence")
        claimed = await self._owned_query(
            self._owned_match(("PENDING",)) + """
            SET m.state = 'ROLLING_BACK' RETURN m.state AS state
            """,
        )
        if not claimed:
            retained = await self._marker()
            state = None if retained is None else str(retained.get("state"))
            if state == GuardState.COMPLETE.value:
                self._bind_marker(retained)
                await self._delete_snapshot()
                return False
            if state != GuardState.ROLLING_BACK.value:
                raise GuardError("Graphiti marker cannot enter rollback")
            if str(retained.get("claim_token") or "") != self._claim_token:
                raise GuardError("Graphiti rollback is owned by another claim")
        async with self.fenced_graph_mutation():
            self._observe_owned_recovery_phase("INVENTORY")
            inventories = await self._restoration_inventories()
            node_uuids = await self._snapshot_uuids(inventories[0], snapshot_label=_SNAPSHOT_NODE)
            relationship_uuids = await self._snapshot_uuids(inventories[1], snapshot_label=_SNAPSHOT_RELATIONSHIP)
            self._observe_owned_recovery_phase("DELETE_NEW_RELATIONSHIPS")
            await self._query(
                """
                MATCH (a)-[r]->(b)
                WHERE (a.group_id = $group_id OR b.group_id = $group_id)
                  AND (r.uuid IS NULL OR NOT r.uuid IN $retained_uuids)
                DELETE r
                """,
                group_id=self._group_id,
                retained_uuids=relationship_uuids,
            )
            self._observe_owned_recovery_phase("DELETE_NEW_NODES")
            await self._query(
                f"""
                MATCH (n)
                WHERE n.group_id = $group_id
                  AND NOT n:{_SNAPSHOT_NODE}
                  AND NOT n:{_SNAPSHOT_RELATIONSHIP}
                  AND NOT n:{_MARKER}
                  AND (n.uuid IS NULL OR NOT n.uuid IN $retained_uuids)
                DETACH DELETE n
                """,
                group_id=self._group_id,
                retained_uuids=node_uuids,
            )
            self._observe_owned_recovery_phase("RESTORE")
            await self._restore_properties(inventories=inventories)
            self._observe_owned_recovery_phase("FULL_VERIFY")
            await self.assert_preexisting_unchanged()
            self._observe_owned_recovery_phase("TERMINAL")
            recovered = await self._owned_query(
                self._owned_match(("ROLLING_BACK",)) + """
                SET m.state = 'RECOVERED_AMBIGUOUS', m.recovery_reason = $reason,
                    m.chat_invocations_json = $chat_invocations_json,
                    m.embedding_usage_json = $embedding_usage_json
                REMOVE g.owner_marker_uuid, g.snapshot_id, g.claim_token
                RETURN m.state AS state
                """,
                reason=reason,
                chat_invocations_json=canonical_json_bytes(chat_invocations).decode("utf-8"),
                embedding_usage_json=canonical_json_bytes(embedding_usage).decode("utf-8"),
            )
            if not recovered or _record_value(recovered[0], "state") != "RECOVERED_AMBIGUOUS":
                raise GuardError("Graphiti recovery marker transition did not commit")
            self._snapshot_cleanup_pending = True
        return True

    async def _pages(self, query: str) -> AsyncIterator[list[object]]:
        cursor = ("", "")
        while True:
            records = await self._query(
                query, group_id=self._group_id, snapshot_id=self._snapshot_id,
                cursor_source=cursor[0], cursor_target=cursor[1], limit=_PAGE_TARGET_LIMIT,
            )
            if not records:
                return
            if len(records) > _PAGE_TARGET_LIMIT:
                raise GuardError("Graphiti page exceeds its actual-target bound")
            page: list[object] = []
            page_bytes = 0
            previous = cursor
            for record in records:
                source = _record_value(record, "source_identity")
                target = _record_value(record, "target_identity")
                if not isinstance(source, str) or not source or not isinstance(target, str):
                    raise GuardError("Graphiti page identity is absent")
                identity = (source, target)
                if identity <= previous:
                    raise GuardError("Graphiti page identities are not strictly increasing")
                size = sum(_property_bytes(_record_value(record, key)) for key in (
                    "source_properties", "target_properties", "expected", "actual",
                ))
                if size > _PAGE_PROPERTY_BYTES:
                    raise GuardError("Graphiti target exceeds the full property byte bound")
                if page_bytes + size > _PAGE_PROPERTY_BYTES:
                    break
                page.append(record)
                page_bytes += size
                previous = identity
            yield page
            cursor = previous

    async def _write_pages(self, read_query: str, write_query: str) -> None:
        async for records in self._pages(read_query):
            await self._write_page(write_query, records)

    async def _write_page(self, write_query: str, records: list[object]) -> None:
        page = [{key: _record_value(record, key) for key in (
            "source_identity", "target_identity",
        )} for record in records]
        written = await self._query(write_query, page=page, snapshot_id=self._snapshot_id)
        if not written or _record_value(written[0], "written") != len(page):
            raise GuardError("Graphiti bounded property write lost an actual target")

    async def _snapshot_uuids(
        self, inventory: list[tuple[str, str]], *, snapshot_label: str,
    ) -> list[object]:
        """Read native UUID values once, within the existing retained byte bound."""
        field = {_SNAPSHOT_NODE: "_newsroom_source_uuid",
                 _SNAPSHOT_RELATIONSHIP: "_newsroom_relationship_uuid"}[snapshot_label]
        expected = {source for source, _ in inventory}
        async def consume(transaction: Any) -> list[object]:
            values: list[object] = []
            seen: set[str] = set()
            size = 256 + sum(192 + len(source.encode()) for source in expected)
            records = await transaction.run(
                f"MATCH (s:{snapshot_label} {{_newsroom_snapshot_id:$snapshot_id}}) "
                f"RETURN elementId(s) AS source_identity, s.{field} AS retained_uuid",
                snapshot_id=self._snapshot_id,
            )
            async for record in records:
                if self._owned_recovery_stop_check is not None and len(values) % _PAGE_TARGET_LIMIT == 0:
                    self._owned_recovery_stop_check()
                source, value = _record_value(record, "source_identity"), _record_value(record, "retained_uuid")
                if type(source) is not str or source not in expected or value is None:
                    raise GuardError("Graphiti guard snapshot identity is malformed")
                if source in seen:
                    raise GuardError("Graphiti inventory has a duplicate actual pair")
                size += 192 + _property_bytes(source) + _property_bytes(value)
                if size > _INVENTORY_BYTES:
                    raise GuardError("Graphiti identity inventory exceeds its byte bound")
                seen.add(source)
                values.append(value)
            if seen != expected:
                raise GuardError("Graphiti identity inventory omits a pre-existing target")
            return values
        async with self._driver.session() as session:
            try:
                session._config.fetch_size = 1
                if session._config.fetch_size != 1:
                    raise AttributeError
            except AttributeError:
                raise GuardError("Graphiti UUID inventory requires bounded fetch") from None
            return await session.execute_write(consume)

    async def _identity_inventory(
        self, query: str, *, snapshot_label: str,
    ) -> list[tuple[str, str]]:
        """Stream identities once; reject incomplete or oversized inventories."""
        async def consume(transaction: Any) -> list[tuple[str, str]]:
            count_result = await transaction.run(
                f"MATCH (s:{snapshot_label} {{_newsroom_snapshot_id:$snapshot_id}}) "
                "RETURN count(s) AS snapshot_count", snapshot_id=self._snapshot_id,
            )
            count = _record_value(await count_result.single(strict=True), "snapshot_count")
            if type(count) is not int or count < 0:
                raise GuardError("Graphiti snapshot coverage count is invalid")
            identities: list[tuple[str, str]] = []
            pairs: set[tuple[str, str]] = set()
            covered: set[str] = set()
            size = 256
            records = await transaction.run(query, snapshot_id=self._snapshot_id)
            async for record in records:
                if self._owned_recovery_stop_check is not None and len(identities) % _PAGE_TARGET_LIMIT == 0:
                    self._owned_recovery_stop_check()
                source = _record_value(record, "source_identity")
                target = _record_value(record, "target_identity")
                if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
                    raise GuardError("Graphiti inventory identity is absent")
                pair = (source, target)
                if pair in pairs:
                    raise GuardError("Graphiti inventory has a duplicate actual pair")
                # Includes retained strings, pair/list/set overhead and coverage.
                size += 320 + len(source.encode()) + len(target.encode())
                if source not in covered:
                    size += 192 + len(source.encode())
                if size > _INVENTORY_BYTES:
                    raise GuardError("Graphiti identity inventory exceeds its byte bound")
                identities.append(pair)
                pairs.add(pair)
                covered.add(source)
            if len(covered) != count:
                raise GuardError("Graphiti identity inventory omits a pre-existing target")
            identities.sort()
            return identities
        async with self._driver.session() as session:
            return await session.execute_write(consume)

    async def _inventory_pages(
        self, inventory: list[tuple[str, str]], query: str,
    ) -> AsyncIterator[list[object]]:
        for start in range(0, len(inventory), _PAGE_TARGET_LIMIT):
            expected = inventory[start:start + _PAGE_TARGET_LIMIT]
            identities = [{"source_identity": source, "target_identity": target}
                          for source, target in expected]
            records = await self._query(query, identities=identities, snapshot_id=self._snapshot_id)
            actual = [(_record_value(row, "source_identity"), _record_value(row, "target_identity"))
                      for row in records]
            if actual != expected:
                raise GuardError("Graphiti property read lost an inventoried actual pair")
            page: list[object] = []
            size = 0
            for row in records:
                row_size = sum(_property_bytes(_record_value(row, key)) for key in (
                    "source_properties", "target_properties", "expected", "actual",
                ))
                if row_size > _PAGE_PROPERTY_BYTES:
                    raise GuardError("Graphiti target exceeds the full property byte bound")
                if page and size + row_size > _PAGE_PROPERTY_BYTES:
                    yield page
                    page, size = [], 0
                page.append(row)
                size += row_size
            if page:
                yield page

    async def _restoration_inventories(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        nodes = await self._identity_inventory(
            f"""
            MATCH (s:{_SNAPSHOT_NODE} {{_newsroom_snapshot_id:$snapshot_id}})
            MATCH (n {{uuid: s._newsroom_source_uuid}})
            WHERE NOT n:{_SNAPSHOT_NODE} AND NOT n:{_SNAPSHOT_RELATIONSHIP} AND NOT n:{_MARKER}
            RETURN elementId(s) AS source_identity, elementId(n) AS target_identity
            """, snapshot_label=_SNAPSHOT_NODE,
        )
        relationships = await self._identity_inventory(
            f"""
            MATCH (s:{_SNAPSHOT_RELATIONSHIP} {{_newsroom_snapshot_id:$snapshot_id}})
            MATCH (a {{uuid: s._newsroom_source_uuid}})
                  -[r {{uuid: s._newsroom_relationship_uuid}}]->(b {{uuid: s._newsroom_target_uuid}})
            WHERE type(r) = s._newsroom_relationship_type
              AND NOT a:{_SNAPSHOT_NODE} AND NOT b:{_SNAPSHOT_NODE}
              AND NOT a:{_SNAPSHOT_RELATIONSHIP} AND NOT b:{_SNAPSHOT_RELATIONSHIP}
              AND NOT a:{_MARKER} AND NOT b:{_MARKER}
            RETURN elementId(s) AS source_identity, elementId(r) AS target_identity
            """, snapshot_label=_SNAPSHOT_RELATIONSHIP,
        )
        return nodes, relationships

    async def _restore_properties(
        self, *, inventories: tuple[list[tuple[str, str]], list[tuple[str, str]]] | None = None,
    ) -> None:
        nodes, relationships = await self._restoration_inventories() if inventories is None else inventories
        async for records in self._inventory_pages(
            nodes, f"""
            UNWIND $identities AS row
            MATCH (s:{_SNAPSHOT_NODE}) WHERE elementId(s) = row.source_identity
              AND s._newsroom_snapshot_id = $snapshot_id
            MATCH (n) WHERE elementId(n) = row.target_identity AND n.uuid = s._newsroom_source_uuid
              AND NOT n:{_SNAPSHOT_NODE} AND NOT n:{_SNAPSHOT_RELATIONSHIP} AND NOT n:{_MARKER}
            RETURN elementId(s) AS source_identity, elementId(n) AS target_identity,
                   properties(s) AS source_properties, properties(n) AS target_properties,
                   s._newsroom_source_labels AS expected, labels(n) AS actual
            ORDER BY source_identity, target_identity
            """,
        ):
            await self._write_page(
                f"""
            UNWIND $page AS row
            MATCH (s:{_SNAPSHOT_NODE}) WHERE elementId(s) = row.source_identity
              AND s._newsroom_snapshot_id = $snapshot_id
            MATCH (n) WHERE elementId(n) = row.target_identity
              AND NOT n:{_SNAPSHOT_NODE} AND NOT n:{_SNAPSHOT_RELATIONSHIP} AND NOT n:{_MARKER}
            SET n = properties(s)
            REMOVE n._newsroom_snapshot_id, n._newsroom_source_uuid, n._newsroom_source_labels
            RETURN count(n) AS written
                """, records,
            )
            await self._restore_page_labels(records)
        async for records in self._inventory_pages(
            relationships, f"""
            UNWIND $identities AS row
            MATCH (s:{_SNAPSHOT_RELATIONSHIP}) WHERE elementId(s) = row.source_identity
              AND s._newsroom_snapshot_id = $snapshot_id
            MATCH (a)-[r]->(b) WHERE elementId(r) = row.target_identity
              AND r.uuid = s._newsroom_relationship_uuid AND type(r) = s._newsroom_relationship_type
              AND a.uuid = s._newsroom_source_uuid AND b.uuid = s._newsroom_target_uuid
              AND NOT a:{_SNAPSHOT_NODE} AND NOT b:{_SNAPSHOT_NODE}
              AND NOT a:{_SNAPSHOT_RELATIONSHIP} AND NOT b:{_SNAPSHOT_RELATIONSHIP}
              AND NOT a:{_MARKER} AND NOT b:{_MARKER}
            RETURN elementId(s) AS source_identity, elementId(r) AS target_identity,
                   properties(s) AS source_properties, properties(r) AS target_properties
            ORDER BY source_identity, target_identity
            """,
        ):
            await self._write_page(
                f"""
                UNWIND $page AS row
                MATCH (s:{_SNAPSHOT_RELATIONSHIP}) WHERE elementId(s) = row.source_identity
                  AND s._newsroom_snapshot_id = $snapshot_id
                MATCH ()-[r]->() WHERE elementId(r) = row.target_identity
                SET r = properties(s)
                REMOVE r._newsroom_snapshot_id, r._newsroom_relationship_uuid,
                       r._newsroom_source_uuid, r._newsroom_target_uuid, r._newsroom_relationship_type
                RETURN count(r) AS written
                """, records,
            )

    async def _restore_page_labels(self, records: list[object]) -> None:
        for record in records:
            target = str(_record_value(record, "target_identity") or "")
            expected = {str(item) for item in (_record_value(record, "expected") or [])}
            actual = {str(item) for item in (_record_value(record, "actual") or [])}
            if not target or any(_LABEL.fullmatch(item) is None for item in expected | actual):
                raise GuardError("Graphiti generation contains an unsafe dynamic label")
            for operation, labels in (("REMOVE", actual - expected), ("SET", expected - actual)):
                for label in sorted(labels):
                    written = await self._query(
                        f"""
                        MATCH (n) WHERE elementId(n) = $target_identity
                          AND NOT n:{_SNAPSHOT_NODE} AND NOT n:{_SNAPSHOT_RELATIONSHIP} AND NOT n:{_MARKER}
                        {operation} n:`{label}` RETURN count(n) AS written
                        """,
                        target_identity=target,
                    )
                    if not written or _record_value(written[0], "written") != 1:
                        raise GuardError("Graphiti label repair lost an actual target")

    async def assert_preexisting_unchanged(self) -> None:
        def validate_node(record: object) -> None:
            snapshot = _record_value(record, "snapshot")
            current = _record_value(record, "current")
            if not isinstance(snapshot, dict) or not isinstance(current, dict):
                raise GuardError("a pre-existing Graphiti node is missing")
            expected = {
                str(key): value
                for key, value in snapshot.items()
                if not str(key).startswith(_RESERVED_PREFIX)
            }
            expected_labels = {
                str(item) for item in snapshot.get("_newsroom_source_labels", [])
            }
            current_labels = {
                str(item) for item in (_record_value(record, "current_labels") or [])
            }
            if _normalise(expected) != _normalise(current) or expected_labels != current_labels:
                raise GuardError("a pre-existing Graphiti node changed across the attempt")

        await self._stream_query(
            f"""
            MATCH (s:{_SNAPSHOT_NODE} {{_newsroom_snapshot_id: $snapshot_id}})
            MATCH (n {{uuid: s._newsroom_source_uuid}})
            WHERE NOT n:{_SNAPSHOT_NODE}
              AND NOT n:{_SNAPSHOT_RELATIONSHIP}
              AND NOT n:{_MARKER}
            RETURN elementId(s) AS snapshot_identity,
                   properties(s) AS snapshot,
                   properties(n) AS current,
                   labels(n) AS current_labels
            """,
            validate_node,
            snapshot_label=_SNAPSHOT_NODE,
            snapshot_id=self._snapshot_id,
        )

        def validate_relationship(record: object) -> None:
            snapshot = _record_value(record, "snapshot")
            current = _record_value(record, "current")
            if not isinstance(snapshot, dict) or not isinstance(current, dict):
                raise GuardError("a pre-existing Graphiti relationship is missing")
            expected = {
                str(key): value
                for key, value in snapshot.items()
                if not str(key).startswith(_RESERVED_PREFIX)
            }
            if (
                _normalise(expected) != _normalise(current)
                or str(_record_value(record, "source_uuid"))
                != str(snapshot.get("_newsroom_source_uuid"))
                or str(_record_value(record, "target_uuid"))
                != str(snapshot.get("_newsroom_target_uuid"))
                or str(_record_value(record, "relationship_type"))
                != str(snapshot.get("_newsroom_relationship_type"))
            ):
                raise GuardError(
                    "a pre-existing Graphiti relationship changed across the attempt"
                )

        await self._stream_query(
            f"""
            MATCH (s:{_SNAPSHOT_RELATIONSHIP} {{_newsroom_snapshot_id: $snapshot_id}})
            MATCH (a {{uuid: s._newsroom_source_uuid}})
                  -[r {{uuid: s._newsroom_relationship_uuid}}]->
                  (b {{uuid: s._newsroom_target_uuid}})
            WHERE type(r) = s._newsroom_relationship_type
              AND NOT a:{_SNAPSHOT_NODE} AND NOT b:{_SNAPSHOT_NODE}
              AND NOT a:{_SNAPSHOT_RELATIONSHIP}
              AND NOT b:{_SNAPSHOT_RELATIONSHIP}
              AND NOT a:{_MARKER} AND NOT b:{_MARKER}
            RETURN elementId(s) AS snapshot_identity,
                   properties(s) AS snapshot,
                   properties(r) AS current,
                   a.uuid AS source_uuid,
                   b.uuid AS target_uuid,
                   type(r) AS relationship_type
            """,
            validate_relationship,
            snapshot_label=_SNAPSHOT_RELATIONSHIP,
            snapshot_id=self._snapshot_id,
        )

    async def complete(self, raw: dict[str, object]) -> None:
        raw_bytes = canonical_json_bytes(raw)
        async with self._generation_fence(("PENDING",)):
            completed = await self._owned_query(
                self._owned_match(("PENDING",)) + """
                SET m.state = 'COMPLETE', m.validated_raw_json = $validated_raw_json,
                    m.validated_raw_digest = $validated_raw_digest,
                    m.provider_attempt_number = $provider_attempt_number
                REMOVE g.owner_marker_uuid, g.snapshot_id, g.claim_token
                RETURN m.state AS state
                """,
                validated_raw_json=raw_bytes.decode("utf-8"),
                validated_raw_digest=digest_bytes(raw_bytes),
                provider_attempt_number=int(raw["provider_attempt_number"]),
            )
            if not completed or _record_value(completed[0], "state") != "COMPLETE":
                raise GuardError("Graphiti completion marker transition did not commit")
            self._snapshot_cleanup_pending = True

    async def completed_raw(self) -> dict[str, object]:
        raw = await self.completed_raw_or_none()
        if raw is None:
            raise GuardError("Graphiti completion marker is absent")
        return raw

    async def completed_raw_or_none(self) -> dict[str, object] | None:
        """Read a matching completed result without creating a mutation marker."""

        marker = await self._marker()
        if marker is None or str(marker.get("state")) != GuardState.COMPLETE.value:
            return None
        self._bind_marker(marker)
        raw_json = marker.get("validated_raw_json")
        retained_digest = marker.get("validated_raw_digest")
        if not isinstance(raw_json, str) or not isinstance(retained_digest, str):
            raise GuardError("Graphiti completion marker has no validated result")
        try:
            raw = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise GuardError("Graphiti completion snapshot is malformed") from exc
        if not isinstance(raw, dict):
            raise GuardError("Graphiti completion snapshot is malformed")
        raw_bytes = canonical_json_bytes(raw)
        if raw_bytes.decode("utf-8") != raw_json or digest_bytes(raw_bytes) != retained_digest:
            raise GuardError("Graphiti completion snapshot digest differs")
        return raw

    async def _delete_snapshot(self) -> None:
        with _guard_phase("SNAPSHOT_CLEANUP", episode_id=self._episode_uuid, attempt_number=self._attempt_number):
            # Each exact snapshot label has its own property index. A malformed
            # dual-labelled snapshot is deleted once, by the first matching label.
            for label in (_SNAPSHOT_NODE, _SNAPSHOT_RELATIONSHIP):
                while True:
                    records = await self._query(
                        f"""
                        MATCH (s:{label} {{_newsroom_snapshot_id: $snapshot_id}})
                        WITH s LIMIT $limit
                        DELETE s RETURN count(*) AS deleted
                        """,
                        snapshot_id=self._snapshot_id, limit=_PAGE_TARGET_LIMIT,
                    )
                    count = None if not records else _record_value(records[0], "deleted")
                    if type(count) is not int or not 0 <= count <= _PAGE_TARGET_LIMIT:
                        raise GuardError("Graphiti bounded snapshot deletion count is invalid")
                    if count < _PAGE_TARGET_LIMIT:
                        break


__all__ = [
    "GuardError",
    "GuardMarker",
    "GuardState",
    "Neo4jMutationGuard",
]
