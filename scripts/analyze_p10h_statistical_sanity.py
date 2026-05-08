#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED = REPO_ROOT / "tables" / "curated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def main() -> int:
    feedback_path = CURATED / "feedback_error_audit.csv"
    sanity_path = CURATED / "statistical_sanity.csv"
    cost_path = CURATED / "cost_quality_reframing.csv"
    required = [feedback_path, sanity_path, cost_path]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing curated statistical inputs: " + ", ".join(missing))

    feedback = by_key(read_csv(feedback_path), "policy")
    cost = by_key(read_csv(cost_path), "policy")
    sanity = read_csv(sanity_path)

    vop = feedback["value_of_profile"]
    periodic = feedback["periodic_profile"]
    random_policy = feedback["random_profile"]
    uncertainty = feedback["uncertainty_only"]
    always = cost["always_profile"]
    vop_cost = cost["value_of_profile"]

    print("P10H statistical sanity summary")
    print("=" * 32)
    print("Feedback error:")
    print(f"- VoP: {as_float(vop, 'feedback_error'):.6f}")
    print(f"- periodic: {as_float(periodic, 'feedback_error'):.6f}")
    print(f"- random: {as_float(random_policy, 'feedback_error'):.6f}")
    print(f"- uncertainty-only: {as_float(uncertainty, 'feedback_error'):.6f}")
    print()
    print("VoP minus baseline feedback-error deltas:")
    for baseline in ["periodic_profile", "random_profile", "uncertainty_only"]:
        row = feedback[baseline]
        print(f"- {baseline}: {as_float(row, 'vop_minus_policy_abs'):.9f}")
    print()
    print("Cost-quality framing:")
    print(f"- always-profile full profiles: {as_float(always, 'full_profile_calls'):.0f}")
    print(f"- VoP full profiles: {as_float(vop_cost, 'full_profile_calls'):.0f}")
    print(f"- VoP full-profile reduction: {as_float(vop_cost, 'full_profile_reduction_vs_always'):.1%}")
    print(f"- always-profile method wall-clock: {as_float(always, 'method_wall_clock_sec'):.2f}s")
    print(f"- VoP method wall-clock: {as_float(vop_cost, 'method_wall_clock_sec'):.2f}s")
    print(f"- VoP task88-excluded speedup: {as_float(vop_cost, 'geomean_speedup_without_task88'):.4f}x")
    print()
    print("Bootstrap / paired sanity checks:")
    for row in sanity:
        print(
            f"- {row['comparison']}: mean={float(row['mean_diff']):.9f}, "
            f"CI=[{float(row['bootstrap_ci_low']):.9f}, {float(row['bootstrap_ci_high']):.9f}], "
            f"wins/losses/ties={row['wins']}/{row['losses']}/{row['ties']}, "
            f"interpretation={row['interpretation']}"
        )
    print()
    print("Paper-facing interpretation:")
    print("- Do not claim statistically significant feedback-error superiority.")
    print("- Treat uncertainty-only as tied or nearly tied with VoP.")
    print("- Main claim: comparable feedback quality with lower profile cost and lower method wall-clock.")
    print("- Search-quality/speedup is useful secondary evidence, especially without task88.")
    print()
    print("Curated source files:")
    for path in required:
        print(f"- {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
