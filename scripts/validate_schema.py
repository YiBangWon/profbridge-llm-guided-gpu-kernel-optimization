from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.data.schema import validate_profile_pair


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ProfBridge profile-pair JSON record.")
    parser.add_argument("--input", required=True, help="Path to a JSON object or JSONL file.")
    args = parser.parse_args()
    path = Path(args.input)
    text = path.read_text(encoding="utf-8").strip()
    records = []
    if text.startswith("{"):
        records = [json.loads(text)]
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    failed = 0
    for idx, record in enumerate(records):
        ok, errors = validate_profile_pair(record)
        print(f"record {idx}: {'PASS' if ok else 'FAIL'}")
        for error in errors:
            print(f"  - {error}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
