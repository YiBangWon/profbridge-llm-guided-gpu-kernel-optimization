#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE = "phase_09h"
POLICIES = [
    "always_profile",
    "cheap_only",
    "periodic_profile",
    "random_profile",
    "uncertainty_only",
    "predicted_benefit_only",
    "value_of_profile",
    "oracle_vop_diagnostic",
]
COMPARE = ["periodic_profile", "random_profile", "uncertainty_only"]


def read_json(path: str):
    with (ROOT / path).open() as f:
        return json.load(f)


def read_jsonl(path: str):
    rows = []
    with (ROOT / path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: str):
    with (ROOT / path).open(newline="") as f:
        return list(csv.DictReader(f))


def write(path: str, text: str):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n")


def write_csv(path: str, rows: list[dict], fields: list[str] | None = None):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(x: float | str | None, digits: int = 6) -> str:
    if x is None:
        return "missing"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v) >= 100:
        return f"{v:.2f}"
    return f"{v:.{digits}f}"


def geomean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v > 0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def binom_p_two_sided(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * prob)


def bootstrap_diff(task_diffs: dict[int, float], iterations: int = 10000, seed: int = 90210):
    tasks = sorted(task_diffs)
    rng = random.Random(seed)
    if not tasks:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    means = []
    for _ in range(iterations):
        sample = [task_diffs[rng.choice(tasks)] for _ in tasks]
        means.append(sum(sample) / len(sample))
    obs = sum(task_diffs.values()) / len(task_diffs)
    return {
        "mean": obs,
        "ci_low": percentile(means, 0.025),
        "ci_high": percentile(means, 0.975),
        "n": len(tasks),
    }


summary = read_json("results/search/phase_10h/aggregate_analysis.json")
closed_loop_summary = read_json("results/search/phase_10h/closed_loop_summary.json")
eval_rows = read_jsonl("results/search/phase_10h/closed_loop_eval.jsonl")
policy_rows = read_csv("tables/phase_10h/table_closed_loop_policy_summary.csv")
policy = {row["policy"]: row for row in policy_rows}
tasks = sorted({int(r["task_id"]) for r in eval_rows})

by_policy_task: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
for row in eval_rows:
    by_policy_task[row["policy"]][int(row["task_id"])].append(row)

task_metrics: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
for pol, by_task in by_policy_task.items():
    for task, rows in by_task.items():
        feedback_vals = [float(r.get("feedback_error_contribution") or 0.0) for r in rows]
        speed_vals = [float(r.get("speedup_over_eager") or 0.0) for r in rows if r.get("correctness_pass")]
        task_metrics[pol][task] = {
            "feedback_error": sum(feedback_vals) / len(feedback_vals),
            "speedup_geomean": geomean(speed_vals) or float("nan"),
            "speedup_best": max(speed_vals) if speed_vals else float("nan"),
            "candidate_count": len(rows),
        }

feedback_audit_rows = []
vop_error = float(policy["value_of_profile"]["feedback_error"])
for pol in POLICIES:
    row = policy[pol]
    err = float(row["feedback_error"])
    abs_diff = vop_error - err
    rel_diff = abs_diff / err if err else None
    feedback_audit_rows.append(
        {
            "policy": pol,
            "feedback_error": err,
            "vop_minus_policy_abs": abs_diff,
            "vop_minus_policy_rel": rel_diff,
            "method_full_profile_calls": int(float(row["method_full_profile_calls"])),
            "method_wall_clock_sec": float(row["method_wall_clock_sec"]),
            "geomean_speedup_without_task88": float(row["geomean_speedup_without_task88"]),
        }
    )

per_task_lines = []
for task in tasks:
    vals = []
    for pol in ["value_of_profile", "periodic_profile", "random_profile", "uncertainty_only"]:
        vals.append(f"{pol}={fmt(task_metrics[pol][task]['feedback_error'], 6)}")
    per_task_lines.append(f"- task {task}: " + ", ".join(vals))

write_csv(
    f"tables/{PHASE}/table_feedback_error_audit.csv",
    feedback_audit_rows,
    [
        "policy",
        "feedback_error",
        "vop_minus_policy_abs",
        "vop_minus_policy_rel",
        "method_full_profile_calls",
        "method_wall_clock_sec",
        "geomean_speedup_without_task88",
    ],
)

write(
    "reports/phase_09h_stat_feedback_error_audit.md",
    f"""# P09H-STAT Feedback Error Audit

## Policy-Level Feedback Error
- VoP: {fmt(policy['value_of_profile']['feedback_error'], 6)}
- Periodic: {fmt(policy['periodic_profile']['feedback_error'], 6)}
- Random: {fmt(policy['random_profile']['feedback_error'], 6)}
- Uncertainty-only: {fmt(policy['uncertainty_only']['feedback_error'], 6)}
- Always-profile: {fmt(policy['always_profile']['feedback_error'], 6)}

## Absolute Differences: VoP Minus Baseline
- VoP - periodic: {fmt(vop_error - float(policy['periodic_profile']['feedback_error']), 9)}
- VoP - random: {fmt(vop_error - float(policy['random_profile']['feedback_error']), 9)}
- VoP - uncertainty-only: {fmt(vop_error - float(policy['uncertainty_only']['feedback_error']), 9)}

Negative values mean VoP has lower feedback error. The magnitudes are on the order of 1e-5 to 1e-4, which is tiny relative to the metric scale near 0.5.

## Relative Differences: VoP Minus Baseline
- VoP vs periodic: {fmt((vop_error - float(policy['periodic_profile']['feedback_error'])) / float(policy['periodic_profile']['feedback_error']) * 100, 6)}%
- VoP vs random: {fmt((vop_error - float(policy['random_profile']['feedback_error'])) / float(policy['random_profile']['feedback_error']) * 100, 6)}%
- VoP vs uncertainty-only: {fmt((vop_error - float(policy['uncertainty_only']['feedback_error'])) / float(policy['uncertainty_only']['feedback_error']) * 100, 6)}%

## Per-Task Feedback Error Contributions
{chr(10).join(per_task_lines)}

## Metric Sensitivity
The feedback-error metric appears saturated or insensitive in this P10H stream: budgeted policies cluster tightly around 0.4999, while cheap-only is near 0.9999 and always-profile is exactly 0 by construction. This supports a cost-quality or non-inferiority framing, not a superiority claim.

## Can This Support a Superiority Claim?
No. P10H does not support wording such as "VoP significantly beats periodic/random in feedback error." The paper-facing wording should be:

> VoP achieves comparable feedback error to budgeted baselines while reducing method full-profile calls and method wall-clock relative to always-profile.

## Source Files
- `results/search/phase_10h/closed_loop_eval.jsonl`
- `results/search/phase_10h/aggregate_analysis.json`
- `tables/phase_10h/table_closed_loop_policy_summary.csv`
""",
)


stat_rows = []
stat_json: dict[str, dict] = {}
for other in COMPARE:
    diffs_feedback = {}
    diffs_speed = {}
    diffs_speed_no88 = {}
    for task in tasks:
        if task in task_metrics["value_of_profile"] and task in task_metrics[other]:
            diffs_feedback[task] = (
                task_metrics["value_of_profile"][task]["feedback_error"]
                - task_metrics[other][task]["feedback_error"]
            )
            diffs_speed[task] = (
                task_metrics["value_of_profile"][task]["speedup_geomean"]
                - task_metrics[other][task]["speedup_geomean"]
            )
            if task != 88:
                diffs_speed_no88[task] = diffs_speed[task]
    for metric, diffs, favorable in [
        ("feedback_error_vop_minus_" + other, diffs_feedback, "lower"),
        ("speedup_without_task88_vop_minus_" + other, diffs_speed_no88, "higher"),
    ]:
        boot = bootstrap_diff(diffs)
        if favorable == "lower":
            wins = sum(1 for d in diffs.values() if d < 0)
            losses = sum(1 for d in diffs.values() if d > 0)
        else:
            wins = sum(1 for d in diffs.values() if d > 0)
            losses = sum(1 for d in diffs.values() if d < 0)
        ties = len(diffs) - wins - losses
        p = binom_p_two_sided(wins, losses)
        row = {
            "comparison": metric,
            "n_tasks": boot["n"],
            "mean_diff": boot["mean"],
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "favorable_direction": favorable,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "sign_test_p_two_sided": p,
            "interpretation": "tiny_or_inconclusive" if metric.startswith("feedback") else "directional_small_n",
        }
        stat_rows.append(row)
        stat_json[metric] = row

write_csv(
    f"tables/{PHASE}/table_statistical_sanity.csv",
    stat_rows,
    [
        "comparison",
        "n_tasks",
        "mean_diff",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "favorable_direction",
        "wins",
        "losses",
        "ties",
        "sign_test_p_two_sided",
        "interpretation",
    ],
)
write(f"results/search/{PHASE}/statistical_sanity.json", json.dumps(stat_json, indent=2, sort_keys=True))

write(
    "reports/phase_09h_stat_bootstrap_sanity.md",
    """# P09H-STAT Bootstrap and Paired Sanity Analysis

## Method
- Unit of resampling: task.
- Feedback metric: mean `feedback_error_contribution` per task and policy.
- Speedup metric: per-task geomean candidate speedup, with task88 excluded for the main speedup sanity check.
- Bootstrap: 10,000 task-resampling replicates with fixed seed.
- Sign test: exact two-sided binomial test on task wins/losses, reported only as a small-n sanity check.

## Results
"""
    + "\n".join(
        f"- `{r['comparison']}`: mean diff {fmt(r['mean_diff'], 9)}, 95% bootstrap CI [{fmt(r['bootstrap_ci_low'], 9)}, {fmt(r['bootstrap_ci_high'], 9)}], wins/losses/ties {r['wins']}/{r['losses']}/{r['ties']}, sign-test p {fmt(r['sign_test_p_two_sided'], 4)}."
        for r in stat_rows
    )
    + """

## Interpretation
The feedback-error differences are tiny and bootstrap intervals overlap zero in this five-task run. This should not be described as statistical superiority. The task88-excluded speedup/search-quality comparisons are more directionally favorable versus periodic/random, but uncertainty-only is close and the sample is too small for strong inference.

## Recommended Wording
- "Comparable feedback error under lower method profile cost."
- "Directionally better task88-excluded search-quality than periodic/random in this run."
- "Uncertainty-only is nearly tied and remains a strong comparator."
- "Small-n bootstrap sanity checks do not support formal superiority claims."
""",
)


non_rows = []
for other in COMPARE:
    base = float(policy[other]["feedback_error"])
    delta = vop_error - base
    rel = delta / base if base else 0.0
    for margin in [0.001, 0.005, 0.01]:
        non_rows.append(
            {
                "baseline": other,
                "margin_type": "absolute",
                "margin": margin,
                "vop_minus_baseline": delta,
                "satisfied": delta <= margin,
            }
        )
    for margin in [0.01, 0.05, 0.10]:
        non_rows.append(
            {
                "baseline": other,
                "margin_type": "relative",
                "margin": margin,
                "vop_minus_baseline": rel,
                "satisfied": rel <= margin,
            }
        )
write_csv(f"tables/{PHASE}/table_noninferiority_margins.csv", non_rows)

write(
    "reports/phase_09h_stat_noninferiority_framing.md",
    f"""# P09H-STAT Non-Inferiority and Equivalence Framing

## Goal
The paper should not claim VoP feedback-error superiority. A safer framing is non-inferiority or practical equivalence: VoP preserves comparable feedback quality while reducing method profile cost.

## Margins Checked
- Absolute feedback-error delta margins: 0.001, 0.005, 0.01.
- Relative feedback-error delta margins: 1%, 5%, 10%.

## Results
VoP satisfies all checked post-hoc margins against periodic, random, and uncertainty-only because VoP is slightly lower on aggregate feedback error:
- VoP - periodic: {fmt(vop_error - float(policy['periodic_profile']['feedback_error']), 9)}
- VoP - random: {fmt(vop_error - float(policy['random_profile']['feedback_error']), 9)}
- VoP - uncertainty-only: {fmt(vop_error - float(policy['uncertainty_only']['feedback_error']), 9)}

## Caveat
These margins are post-hoc and should be presented as sensitivity analysis, not as a formal non-inferiority proof. The small task count and saturated metric make formal inference weak.

## Recommended Paper Wording
- "VoP achieves comparable feedback error to budgeted baselines in this run."
- "The feedback-error differences are statistically indistinguishable in this small run."
- "The main benefit is lower method cost, not feedback-error superiority."
""",
)


cost_rows = []
for pol in ["always_profile", "value_of_profile", "periodic_profile", "random_profile", "uncertainty_only"]:
    row = policy[pol]
    cost_rows.append(
        {
            "policy": pol,
            "full_profile_calls": row["method_full_profile_calls"],
            "full_profile_reduction_vs_always": row["full_profile_reduction_vs_always"],
            "method_wall_clock_sec": row["method_wall_clock_sec"],
            "feedback_error": row["feedback_error"],
            "geomean_speedup": row["geomean_speedup"],
            "geomean_speedup_without_task88": row["geomean_speedup_without_task88"],
            "claim_support": (
                "quality_upper_bound"
                if pol == "always_profile"
                else "main_cost_quality_policy"
                if pol == "value_of_profile"
                else "budgeted_baseline"
            ),
        }
    )
write_csv(f"tables/{PHASE}/table_cost_quality_reframing.csv", cost_rows)
write_csv(f"results/search/{PHASE}/figure_data_cost_quality_reframing.csv", cost_rows)

write(
    "reports/phase_09h_stat_cost_quality_reframing.md",
    f"""# P09H-STAT Cost-Quality and Pareto Reframing

## Supported Claims
- Cost reduction versus always-profile: VoP uses {policy['value_of_profile']['method_full_profile_calls']}/10 method full profiles, a 50% reduction.
- Wall-clock reduction versus always-profile: VoP {fmt(policy['value_of_profile']['method_wall_clock_sec'], 2)}s versus always-profile {fmt(policy['always_profile']['method_wall_clock_sec'], 2)}s.
- Comparable feedback quality versus budgeted baselines: VoP, periodic, random, and uncertainty-only are tightly clustered around 0.4999 feedback error.
- Task88-excluded search-quality versus periodic/random: VoP {fmt(policy['value_of_profile']['geomean_speedup_without_task88'], 4)}x, periodic {fmt(policy['periodic_profile']['geomean_speedup_without_task88'], 4)}x, random {fmt(policy['random_profile']['geomean_speedup_without_task88'], 4)}x.

## Unsupported Claims
- VoP feedback-error superiority.
- Clear VoP superiority over uncertainty-only.
- ProfileSketch prompt-representation superiority.
- Consistently faster kernels.

## Recommended Figure Framing
Plot full-profile calls or method wall-clock on the x-axis and feedback/search-quality on the y-axis. The visual message should be cost-quality tradeoff, not statistical dominance.
""",
)


addendum_text = """# P09H-STAT Advisor Addendum

## What Changed After Statistical Sanity Check
P10H remains useful, but the paper-facing interpretation should be more conservative than the raw `strong_acceptance_evidence` label. The feedback-error margin between VoP and periodic/random/uncertainty-only is extremely small, and uncertainty-only is nearly tied.

## Safer Interpretation
- Main claim: 50% full-profile reduction plus lower method wall-clock under comparable feedback quality.
- Secondary claim: task88-excluded search-quality is directionally better than periodic/random.
- Caveated claim: VoP is competitive with uncertainty-only, not clearly superior.
- ProfileSketch should be framed as a structured acquisition/control representation, not as a better prompt representation.

## Exact Caveats
- Feedback-error margins are tiny; do not claim statistical superiority.
- Uncertainty-only is nearly tied with VoP.
- ProfileSketch representation ablation is mixed.
- P10H covers five tasks.
- Task88 is influential and task36 remains unresolved.

## One-Minute Explanation
P10H is still helpful, but not because VoP decisively lowers feedback error. The honest story is that VoP cuts method full-profile calls by 50% and lowers method wall-clock versus always-profile while maintaining feedback error comparable to other budgeted policies. The stronger search-quality signal is task88-excluded speedup versus periodic/random, but uncertainty-only is almost tied, so the paper should center ProfileSketch/VoP as acquisition accounting and control rather than as a clearly superior optimizer.

## What Not To Say
- Do not say "VoP significantly beats periodic/random in feedback error."
- Do not say "VoP dominates uncertainty-only."
- Do not say "ProfileSketch is a better prompt representation."
- Do not say "ProfBridge consistently finds faster kernels."

## Decisions Needed From Advisor
1. Is comparable feedback quality plus lower profile cost enough for the main CGO claim?
2. Should uncertainty-only be framed as an ablation that nearly matches VoP?
3. Should P09I add a formal non-inferiority framing or keep it as sensitivity analysis?
4. Should task36 be fixed before submission?
5. Is the current accept estimate of 40-48% honest enough before advisor feedback?
"""
write("reports/phase_09h_stat_advisor_addendum.md", addendum_text)

advisor_note = """

## P09H-STAT Note
Feedback-error margins are tiny; we do not claim statistical superiority. Uncertainty-only is nearly tied; VoP should be framed as a general ProfileSketch acquisition controller, not clearly superior to uncertainty-only in this run. Main evidence is profile reduction and wall-clock reduction under comparable feedback quality. See `reports/phase_09h_stat_advisor_addendum.md`.
"""
for path in [
    "reports/phase_09h_advisor_one_page_memo.md",
    "reports/phase_09h_advisor_slide_outline.md",
    "reports/phase_09h_advisor_meeting_script.md",
    "reports/phase_09h_reviewer_attack_response_after_p10h.md",
    "reports/phase_09h_summary.md",
]:
    p = ROOT / path
    text = p.read_text()
    if "P09H-STAT Note" not in text:
        p.write_text(text.rstrip() + advisor_note + "\n")


unsafe_re = re.compile(
    r"VoP beats|significantly|strong|superior|dominates|consistently faster|better generator",
    re.IGNORECASE,
)
paper_files = sorted((ROOT / "paper/phase_09f/sections").glob("*.tex")) + [ROOT / "paper/phase_09f/main.tex"]
wording_hits = []
for p in paper_files:
    text = p.read_text()
    for i, line in enumerate(text.splitlines(), 1):
        if unsafe_re.search(line):
            wording_hits.append({"file": str(p.relative_to(ROOT)), "line": i, "text": line.strip()})
    patched = text
    replacements = {
        "The strongest claim is profile-budgeted feedback control": "The safest paper-facing claim is profile-budgeted feedback control",
        "The strongest claim is": "The safest paper-facing claim is",
        "strong baseline": "important baseline",
        "strong comparator": "important comparator",
        "stronger": "more defensible",
        "strong": "carefully bounded",
        "superior": "better in this bounded run",
        "dominates": "is clearly above",
        "consistently faster": "bounded speedup-preserving",
        "better generator": "different feedback controller",
        "significantly": "directionally",
        "VoP beats": "VoP directionally improves over",
    }
    for a, b in replacements.items():
        patched = patched.replace(a, b)
    if patched != text:
        p.write_text(patched)

write(
    "reports/phase_09h_stat_paper_wording_audit.md",
    "# P09H-STAT Paper Wording Audit\n\n"
    + "## Unsafe Wording Searched\n"
    + "- `VoP beats`, `significantly`, `strong`, `superior`, `dominates`, `consistently faster`, `better generator`\n\n"
    + "## Hits Before Patch\n"
    + (
        "\n".join(f"- `{h['file']}:{h['line']}`: {h['text']}" for h in wording_hits)
        if wording_hits
        else "- No hits found."
    )
    + "\n\n## Patch Applied\n"
    + "- Replaced riskier paper-section wording with bounded language where present.\n"
    + "- Added advisor-facing notes that feedback-error margins are tiny and uncertainty-only is nearly tied.\n\n"
    + "## Safe Wording To Use\n"
    + "- VoP achieves comparable feedback error to budgeted baselines while reducing method full-profile calls.\n"
    + "- VoP directionally improves over non-adaptive baselines in our closed-loop run, but margins are small.\n"
    + "- Uncertainty-only is an important baseline and nearly matches VoP.\n"
    + "- ProfileSketch primarily enables acquisition accounting, uncertainty, provenance, and policy decisions rather than universally improving prompt quality.\n",
)


attacks = [
    (
        "Feedback error margin is within noise.",
        "Yes. We no longer claim feedback-error superiority. The supported claim is comparable feedback quality under lower method profile cost.",
        "VoP 0.499866, periodic 0.499941, random 0.499935, uncertainty-only 0.499876.",
        "statistical superiority",
        "cost reduction and wall-clock reduction with comparable error",
    ),
    (
        "Uncertainty-only is tied with VoP.",
        "Yes, nearly. We should present uncertainty-only as an important comparator and say VoP is a general controller rather than clearly superior in this run.",
        "VoP task88-excluded 1.2337x, uncertainty-only 1.2318x.",
        "VoP dominates uncertainty-only",
        "VoP is competitive and more general in policy formulation",
    ),
    (
        "ProfileSketch representation ablation is mixed.",
        "Correct. ProfileSketch is not proven as a better prompt representation; it is a structured acquisition/control representation.",
        "P10H representation table.",
        "ProfileSketch improves prompt quality universally",
        "ProfileSketch supports accounting, uncertainty, provenance, and policy decisions",
    ),
    (
        "The main result is only five tasks.",
        "Correct. We should call it bounded live validation and avoid broad generalization.",
        "P10H tasks: 2, 6, 32, 50, 88.",
        "generalizes to all KernelBench",
        "bounded live evidence before scale-up",
    ),
    (
        "Task88 dominates speedup.",
        "We explicitly report task88-excluded results and keep speedup secondary.",
        "VoP without task88 1.2337x.",
        "consistent faster kernels",
        "task88-excluded search-quality is directionally positive",
    ),
    (
        "Task36 failed.",
        "Task36 remains unresolved and excluded from main claims.",
        "P10D/P10H limitation reports.",
        "ProfBridge handles task36",
        "task36 is a disclosed limitation",
    ),
]
write(
    "reports/phase_09h_stat_top_reviewer_attacks.md",
    "# P09H-STAT Top Reviewer Attacks\n\n"
    + "\n\n".join(
        f"## {i+1}. {q}\n- Reviewer concern: {q}\n- Honest answer: {ans}\n- Evidence: {ev}\n- What we no longer claim: {no}\n- What remains supported: {yes}\n- Possible follow-up if professor asks: run a larger P09I/P10H follow-up or present as limitation."
        for i, (q, ans, ev, no, yes) in enumerate(attacks)
    ),
)


# Update state/progress.yaml with a final P09H-STAT block. Duplicate top-level keys are
# already used in this phase log, so append the latest values for the repo's phase tracker.
progress_path = ROOT / "state/progress.yaml"
progress = progress_path.read_text()
block = """

# P09H-STAT update
current_phase_id: P09H-STAT
current_phase_name: "Statistical Sanity and Claim Recalibration after P10H"
p09h_stat_status: ready_for_advisor_meeting
p10h_raw_status: strong_acceptance_evidence
p10h_paper_facing_interpretation: moderate_evidence_with_strong_cost_reduction_signal
p09h_stat_feedback_error_superiority_supported: false
p09h_stat_vop_vs_uncertainty_interpretation: tied_or_nearly_tied
p09h_stat_main_claim: comparable_feedback_quality_with_lower_profile_cost
p09h_stat_ready_for_advisor_meeting: true
p09h_stat_accept_probability_estimate: "40-48 current; 45-52 after clean draft/advisor feedback"
next_phase_id: P09I
"""
if "# P09H-STAT update" in progress:
    progress = progress.split("# P09H-STAT update")[0].rstrip() + block
else:
    progress = progress.rstrip() + block
progress_path.write_text(progress.rstrip() + "\n")

write(
    "reports/phase_status.md",
    """# Phase Status

Current Phase: P09H-STAT - Statistical Sanity and Claim Recalibration after P10H

## P10H Paper-Facing Reinterpretation
- Raw P10H status: `strong_acceptance_evidence`.
- Paper-facing interpretation: `moderate_evidence_with_strong_cost_reduction_signal`.
- Feedback-error superiority supported: false.
- Main claim: comparable feedback quality with lower profile cost.

## Key Sanity Results
- Feedback-error margins are tiny and should not be described as statistically significant.
- Uncertainty-only is tied or nearly tied with VoP.
- VoP's strongest support is 50% full-profile reduction and lower method wall-clock versus always-profile.
- Task88-excluded search-quality remains useful secondary evidence.

## Advisor Package Status
- Advisor addendum: `reports/phase_09h_stat_advisor_addendum.md`.
- Top reviewer attacks: `reports/phase_09h_stat_top_reviewer_attacks.md`.
- Ready for advisor meeting: true.

## Next Recommended Command
Review `reports/phase_09h_stat_advisor_addendum.md` before the meeting.
""",
)
