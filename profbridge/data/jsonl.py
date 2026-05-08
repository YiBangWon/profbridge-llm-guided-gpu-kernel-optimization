from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return records


def read_many(patterns: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pattern in patterns:
        paths = sorted(glob(pattern))
        if not paths and Path(pattern).exists():
            paths = [pattern]
        for path in paths:
            records.extend(read_jsonl(path))
    return records
