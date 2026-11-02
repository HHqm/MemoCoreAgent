from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

# SQLite caps the number of bound parameters per statement.
_BATCH = 500


class BlobCache:
    """A SQLite key -> bytes table, read and written in batches."""

    def __init__(self, path: Path, table: str):
        if not table.isidentifier():
            raise ValueError(f"bad table name {table!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._table = table
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        self._conn.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        for start in range(0, len(keys), _BATCH):
            batch = list(keys[start : start + _BATCH])
            rows = self._conn.execute(
                f"SELECT key, value FROM {self._table} WHERE key IN ({','.join('?' * len(batch))})", batch
            ).fetchall()
            found.update(rows)
        return found

    def put_many(self, items: Iterable[tuple[str, bytes]]) -> None:
        self._conn.executemany(f"INSERT OR REPLACE INTO {self._table} (key, value) VALUES (?, ?)", list(items))
        self._conn.commit()
