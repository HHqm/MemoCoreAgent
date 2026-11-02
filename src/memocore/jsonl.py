from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A line cut off by an interrupted run; its question is simply redone on resume.
                continue


class JsonlWriter:
    """Thread-safe append-only writer that flushes every record, so an interrupted run loses nothing."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._handle = None

    def __enter__(self) -> "JsonlWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8", newline="\n")
        if _ends_mid_line(self._path):
            self._handle.write("\n")
        return self

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()

    def __exit__(self, *exc_info) -> None:
        self._handle.close()


def _ends_mid_line(path: Path) -> bool:
    if path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) != b"\n"
