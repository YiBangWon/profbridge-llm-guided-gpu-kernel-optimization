#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.profile_sketch.value_of_profile import evaluate_policies


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True)+"\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f) for f in fields})


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(',') if x.strip()]


def main() -> int:
    ap=argparse.ArgumentParser(description="Evaluate Value-of-Profile policies on ProfileSketch records.")
    ap.add_argument('--profile-sketches', default='results/search/phase_10f/profile_sketches.jsonl')
    ap.add_argument('--profile-budget-fraction', default='0.25,0.5,0.75')
    ap.add_argument('--policy', default='always_profile,cheap_only,periodic_profile,random_profile,uncertainty_only,predicted_benefit_only,value_of_profile,oracle_vop')
    ap.add_argument('--online-update', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--out-json', default='results/search/phase_10f/vop_policy_summary.json')
    ap.add_argument('--decisions-out', default='results/search/phase_10f/vop_policy_decisions.jsonl')
    ap.add_argument('--report', default='reports/phase_10f_value_of_profile_eval.md')
    ap.add_argument('--method-report', default='reports/phase_10f_value_of_profile_method.md')
    args=ap.parse_args()
    sketches=read_jsonl(REPO_ROOT/args.profile_sketches)
    budgets=[float(x) for x in parse_csv_list(args.profile_budget_fraction)]
    policies=parse_csv_list(args.policy)
    summaries, decisions=evaluate_policies(sketches, budgets, policies)
    tasks=sorted({int(s['task_id']) for s in sketches})
    out={
        'dry_run': args.dry_run,
        'online_update_requested': args.online_update,
        'device': args.device,
        'profile_sketch_count': len(sketches),
        'task_count': len(tasks),
        'tasks': tasks,
        'budgets': budgets,
        'policies': policies,
        'summaries': summaries,
        'note': 'Offline replay over existing ProfileSketch records; no new API call or NCU run is performed by this evaluator.'
    }
    if not args.dry_run:
        write_json(REPO_ROOT/args.out_json, out)
        write_jsonl(REPO_ROOT/args.decisions_out, decisions)
        fields=['budget_fraction','policy_name','candidate_count','full_profile_call_count','full_profile_reduction_vs_always','feedback_error_shadow_mean','unprofiled_prediction_error_shadow_mean','method_wall_clock_sec_total','eval_only_shadow_profile_time_sec_total','predictor_overhead_sec_total','best_speedup_all_timing_visible','best_speedup_profiled_subset','geomean_speedup_all_timing_visible','geomean_speedup_profiled_subset']
        write_csv(REPO_ROOT/'tables/phase_10f/table_vop_policy_summary.csv', summaries, fields)
        write_csv(REPO_ROOT/'results/search/phase_10f/figure_data_vop_calls_vs_error.csv', summaries, ['budget_fraction','policy_name','full_profile_call_count','full_profile_reduction_vs_always','feedback_error_shadow_mean'])
        write_csv(REPO_ROOT/'results/search/phase_10f/figure_data_vop_speedup_vs_budget.csv', summaries, ['budget_fraction','policy_name','full_profile_call_count','best_speedup_profiled_subset','geomean_speedup_profiled_subset','best_speedup_all_timing_visible'])
    method = """# Phase 10F Value-of-Profile Method\n\nValue-of-Profile is implemented as an explicit acquisition score:\n\n`VoP(c) = uncertainty_score(c) * predicted_benefit(c) / max(profile_cost_estimate(c), epsilon)`\n\nPolicies evaluated: always-profile, cheap-only, periodic, random, uncertainty-only, predicted-benefit-only, Value-of-Profile, and oracle-VoP diagnostic. Oracle-VoP is diagnostic only and is not a ProfBridge method claim. Decisions are made under 25%, 50%, and 75% full-profile budgets.\n"""
    (REPO_ROOT/args.method_report).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT/args.method_report).write_text(method, encoding='utf-8')
    best_50=[s for s in summaries if s['budget_fraction']==0.5]
    best_sorted=sorted(best_50, key=lambda s: (s.get('feedback_error_shadow_mean') is None, s.get('feedback_error_shadow_mean') or 999))
    lines=[
        '# Phase 10F Value-of-Profile Evaluation' if not args.dry_run else '# Phase 10F Value-of-Profile Dry Run',
        '',
        f'- Dry run: `{args.dry_run}`',
        f'- ProfileSketch records: `{len(sketches)}`',
        f'- Tasks: `{tasks}`',
        f'- Budgets: `{budgets}`',
        f'- Policies: `{policies}`',
        f'- Best 50% budget policy by feedback error: `{best_sorted[0]["policy_name"] if best_sorted else "missing"}`',
        '',
        '## 50% Budget Summary',
        '',
        '| Policy | Full profiles | Reduction | Feedback error | Method wall-clock |',
        '|---|---:|---:|---:|---:|',
    ]
    for s in best_50:
        lines.append(f"| {s['policy_name']} | {s['full_profile_call_count']} | {s['full_profile_reduction_vs_always']:.3f} | {s['feedback_error_shadow_mean']:.6f} | {s['method_wall_clock_sec_total']:.2f} |")
    lines += ['', 'Note: this is an offline replay over existing measured ProfileSketch records, not a new live NCU run.']
    (REPO_ROOT/args.report).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(REPO_ROOT/args.report)
    if not args.dry_run:
        print(REPO_ROOT/args.out_json)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
