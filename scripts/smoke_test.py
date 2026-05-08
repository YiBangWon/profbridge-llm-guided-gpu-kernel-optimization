from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.data.schema import example_profile_pair, validate_profile_pair
from profbridge.models.online_predictor import toy_online_update_reduces_error
from profbridge.profile.ptxas import synthetic_parser_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dependency-light ProfBridge smoke checks.")
    parser.parse_args()
    ok_schema, schema_errors = validate_profile_pair(example_profile_pair())
    checks = {
        "schema_example": ok_schema,
        "ptxas_parser": synthetic_parser_test(),
        "online_predictor_toy": toy_online_update_reduces_error(),
    }
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if schema_errors:
        print("schema errors:", schema_errors)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
