"""Expire diagnostic bodies after the CURRENT-state cutover; retain business pins."""
from __future__ import annotations

import sqlite3

from .native_progress_state import current_state_retained_sequences

_BUSINESS_STAGES = frozenset({
    'EMBEDDING_STARTED', 'ASSESSMENT_STARTED', 'PUBLICATION_STARTED',
    'ACKNOWLEDGED', 'COPY_CORRECTION_PREPARED',
})
_KINDS = (
    'NATIVE_REVISION_PROGRESS', 'NATIVE_SOURCE_PORTFOLIO',
    'NATIVE_SERVICE_CYCLE_STARTED', 'NATIVE_SERVICE_CYCLE_TERMINAL',
)


def prune_native_diagnostic_payloads(connection: sqlite3.Connection) -> dict:
    """One explicit maintenance transaction; no boot invocation or ledger deletion.

    Exact source/provider/ACK payloads and ledger identity/header chains stay.
    The caller owns quiescence and any later physical SQLite compaction.
    """
    if connection.in_transaction:
        raise ValueError('diagnostic expiry requires an idle connection')
    connection.execute('BEGIN IMMEDIATE')
    report = {'expired_payloads': 0, 'expired_bytes': 0, 'unclassified_payloads': 0}
    try:
        protected = current_state_retained_sequences(connection)
        cursor = 0
        while True:
            rows = connection.execute(
                'SELECT seq,kind,length(CAST(payload_json AS BLOB)), '
                "CASE WHEN json_valid(payload_json) THEN json_extract(payload_json,'$.stage') END "
                'FROM ledger WHERE seq>? AND kind IN (?,?,?,?) AND payload_json IS NOT NULL '
                'ORDER BY seq LIMIT 256', (cursor, *_KINDS),
            ).fetchall()
            if not rows:
                break
            cursor = rows[-1][0]
            expired = []
            for seq, kind, size, stage in rows:
                if seq in protected or (kind == _KINDS[0] and stage in _BUSINESS_STAGES):
                    continue
                if kind == _KINDS[0] and (type(stage) is not str or not stage):
                    report['unclassified_payloads'] += 1
                    continue
                expired.append((seq,))
                report['expired_bytes'] += size
            before = connection.total_changes
            connection.executemany('UPDATE ledger SET payload_json=NULL WHERE seq=? AND payload_json IS NOT NULL', expired)
            if connection.total_changes-before != len(expired):
                raise ValueError('diagnostic expiry changed an unexpected row')
            report['expired_payloads'] += len(expired)
        if current_state_retained_sequences(connection) != protected:
            raise ValueError('CURRENT roots changed during diagnostic expiry')
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    report['protected_sequences'] = len(protected)
    return report
