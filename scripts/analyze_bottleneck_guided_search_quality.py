#!/usr/bin/env python3
"""Reproduce the headline causal-ablation result from curated artifacts.

The public release ships curated summaries rather than the full private phase
logs (see PUBLIC_RELEASE_CHECKLIST.md). This script reads the curated summary
JSON, prints the per-arm table and contrast tests, and re-renders the figure —
no GPU, no LLM, no private logs required.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "tables/curated/bottleneck_guided_retro_summary.json"


def main() -> int:
    if not SUMMARY.exists():
        print(f"missing curated summary: {SUMMARY}", file=sys.stderr)
        return 1
    data = json.loads(SUMMARY.read_text())

    print(f"raw records (before honesty gate): {data['raw_records']}\n")
    print(f"{'arm':30s} {'n_raw':>5s} {'n_gated':>7s} "
          f"{'geo_vs_eager':>12s} {'geo_vs_static':>13s}")
    for row in sorted(data["per_arm"], key=lambda r: r["arm"]):
        ge = row.get("geomean_speedup_over_eager")
        gc = row.get("geomean_speedup_vs_static_control")
        print(f"{row['arm']:30s} {row['n_raw']:5d} {row['n_gated']:7d} "
              f"{(f'{ge:.4f}' if ge != '' else 'NA'):>12s} "
              f"{(f'{gc:.4f}' if gc != '' else 'NA'):>13s}")

    print("\n== Contrasts (paired by task, honesty-gated) ==")
    for name, c in data["contrasts"].items():
        print(f"  {name:34s} ratio={c['point_ratio']} ci95={c['ci95']} "
              f"p(a_not_better)={c['p_one_sided_a_not_better']} "
              f"(n_tasks={c['n_tasks']})")

    print("\nInterpretation:\n  " + data["interpretation"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping figure re-render)")
        return 0

    order = [
        "control_static", "corrupted_wrong", "corrupted_random",
        "corrupted_noisy", "oracle_measured_bottleneck", "guided_label_only",
        "prompt_only_framing", "guided_predicted_bottleneck",
    ]
    by_arm = {r["arm"]: r for r in data["per_arm"]}
    order = [o for o in order if o in by_arm]
    vals = [by_arm[o]["geomean_speedup_vs_static_control"] for o in order]
    colors = []
    for o in order:
        if o == "control_static":
            colors.append("#888888")
        elif o.startswith("corrupted"):
            colors.append("#d98c5f")
        elif o == "oracle_measured_bottleneck":
            colors.append("#5b8def")
        elif o in ("prompt_only_framing", "guided_label_only"):
            colors.append("#e0b03c")
        else:
            colors.append("#3fae5a")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", lw=1, ls="--", label="static control (=1.0)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
    ax.set_ylabel("Geomean speedup vs static control\n(honesty-gated candidates)")
    ax.set_title(
        "Causal ablation: optimization *framing*, not profiler-derived bottleneck\n"
        "content, explains most of the measured gain (phase 10e+10g, real gpt-5.5)"
    )
    ax.set_ylim(0.95, max(vals) * 1.08)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    figdir = REPO / "figures/curated"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "fig_bottleneck_guided_search_quality.png"
    fig.savefig(out, dpi=150)
    print(f"\nre-rendered {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
