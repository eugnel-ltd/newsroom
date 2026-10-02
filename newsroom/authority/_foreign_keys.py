from __future__ import annotations

from collections import defaultdict
import re
import sqlite3


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'



def foreign_key_children(
    connection: sqlite3.Connection, parent: str, *, key: str | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    """Declared child lookup columns; a selected parent key is a scalar lookup."""
    children = set()
    tables = tuple(row[0] for row in connection.execute(
        "SELECT name FROM main.sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ))
    for table in tables:
        groups = defaultdict(list)
        for row in connection.execute(f"PRAGMA main.foreign_key_list({_identifier(table)})"):
            if row[2] == parent:
                groups[row[0]].append(row)
        for rows in groups.values():
            if key is None:
                children.add((table, tuple(row[3] for row in sorted(rows, key=lambda row: row[1]))))
            else:
                children.update((table, (row[3],)) for row in rows if row[4] == key)
    return sorted(children)


def index_foreign_key_children(
    connection: sqlite3.Connection, parent: str, indexes: list[str], *,
    key: str | None = None, prefix: str = "_fk_maintenance_",
) -> list[tuple[str, tuple[str, ...]]]:
    """Add only missing full-key lookup indexes; caller owns transaction/cleanup.

    Native FK checks remain enabled and authoritative. These covering prefixes
    avoid per-parent full child scans on authenticated matching-type BINARY authority keys.
    Partial/expression/wrong-leading-column indexes are not a full-key lookup.
    Composite equality keys can use an existing prefix in either column order.
    """
    children = foreign_key_children(connection, parent, key=key)
    for table, columns in children:
        existing = tuple(row[1] for row in connection.execute(
            f"PRAGMA main.index_list({_identifier(table)})"
        ) if not row[4])
        if any(
            len(info := tuple(row[2] for row in connection.execute(
                f"PRAGMA main.index_info({_identifier(index)})"
            ))) >= len(columns)
            and set(info[:len(columns)]) == set(columns)
            for index in existing
        ):
            continue
        number = len(indexes)
        name = f"{prefix}{number}"
        while connection.execute("SELECT 1 FROM main.sqlite_schema WHERE name=?", (name,)).fetchone() is not None:
            number += 1
            name = f"{prefix}{number}"
        connection.execute(
            f"CREATE INDEX {_identifier(name)} ON {_identifier(table)}"
            f"({','.join(_identifier(column) for column in columns)})"
        )
        indexes.append(name)
    return children

def has_foreign_key_violation(
    connection: sqlite3.Connection, *, table_names: tuple[str, ...] | None = None,
) -> bool:
    """Check every FK after the caller has authenticated the authority schema.

    Equal, strictly stored types with BINARY collation need no FK affinity
    conversion. Ordered set subtraction scans their covering indexes instead
    of repeatedly fetching wide child rows and random parent pages. A NULL in any
    child key satisfies SQLite's FK rule. Other schemas keep SQLite's checker.
    An optional table selection restricts child tables only; parent schemas and
    every FK of each selected table remain checked. The default checks all tables.
    No validation result survives this call and no data or setting is changed.
    """
    tables = dict(connection.execute(
        "SELECT name,sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ))
    selected = tables if table_names is None else table_names
    if any(name not in tables for name in selected):
        raise ValueError("unknown foreign-key table selection")
    types = {
        name: {row[1]: row[2] for row in connection.execute(
            f"PRAGMA table_info({_identifier(name)})"
        )}
        for name in tables
    }
    eligible = {
        name for name, sql in tables.items()
        if sql.rstrip().upper().endswith("STRICT")
        and re.search(r"\bCOLLATE\b", sql, re.IGNORECASE) is None
    }
    for table in selected:
        groups = defaultdict(list)
        for row in connection.execute(f"PRAGMA foreign_key_list({_identifier(table)})"):
            groups[row[0]].append(row)
        if not groups:
            continue
        if table not in eligible or any(
            row[2] not in eligible
            or types[table].get(row[3]) not in {"TEXT", "INTEGER", "BLOB"}
            or types[table].get(row[3]) != types.get(row[2], {}).get(row[4])
            for rows in groups.values() for row in rows
        ):
            if connection.execute(
                f"PRAGMA foreign_key_check({_identifier(table)})"
            ).fetchone() is not None:
                return True
            continue
        if connection.execute(f"SELECT 1 FROM {_identifier(table)} LIMIT 1").fetchone() is None:
            continue
        for rows in groups.values():
            rows.sort(key=lambda row: row[1])
            child = ",".join(_identifier(row[3]) for row in rows)
            parent = ",".join(_identifier(row[4]) for row in rows)
            nonnull = " AND ".join(f"{_identifier(row[3])} IS NOT NULL" for row in rows)
            order = ",".join(str(index) for index in range(1, len(rows) + 1))
            if connection.execute(
                f"SELECT {child} FROM {_identifier(table)} WHERE {nonnull} "
                f"EXCEPT SELECT {parent} FROM {_identifier(rows[0][2])} "
                f"ORDER BY {order} LIMIT 1"
            ).fetchone() is not None:
                return True
    return False
