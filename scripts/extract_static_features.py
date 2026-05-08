from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.profile.ptxas import synthetic_parser_test
from profbridge.profile.static_features import extract_static_features
from profbridge.utils.env import ensure_dir, utc_timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract cheap static features from a candidate source file.")
    parser.add_argument("--candidate", help="Candidate Python/CUDA source path.")
    parser.add_argument("--output", help="Output JSON path.")
    parser.add_argument("--run-ptxas-parser-test", action="store_true", help="Run synthetic ptxas parser test too.")
    args = parser.parse_args()
    if args.run_ptxas_parser_test:
        print(f"ptxas_parser_test: {'PASS' if synthetic_parser_test() else 'FAIL'}")
    if not args.candidate:
        if args.run_ptxas_parser_test:
            return 0
        parser.error("--candidate is required unless --run-ptxas-parser-test is set")
    features = extract_static_features(args.candidate)
    out = Path(args.output) if args.output else ensure_dir("results/profiles") / f"static_features_{utc_timestamp().replace(':', '').replace('+', 'Z')}.json"
    out.write_text(json.dumps(features, indent=2, sort_keys=True), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
