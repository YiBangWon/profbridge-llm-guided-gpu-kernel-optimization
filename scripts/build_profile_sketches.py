#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.profile_sketch.build import build_sketch

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def main() -> int:
    ap = argparse.ArgumentParser(description="Build ProfileSketch records from profile-pair JSONL.")
    ap.add_argument("--input", default="examples/minimal_profile_pair.jsonl")
    ap.add_argument("--out", default="results/profile_sketches.jsonl")
    args = ap.parse_args()
    rows = read_jsonl(REPO_ROOT / args.input)
    sketches = []
    for row in rows:
        sketch = build_sketch(row, phase="public_example")
        if sketch is not None:
            sketches.append(sketch.to_dict())
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for sketch in sketches:
            handle.write(json.dumps(sketch, sort_keys=True) + "\n")
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
