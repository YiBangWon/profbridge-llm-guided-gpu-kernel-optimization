from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.data.jsonl import append_jsonl
from profbridge.profile.ncu import load_requested_metrics, run_ncu_profile
from profbridge.utils.env import collect_environment_metadata, ensure_dir, utc_timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile a KernelBench baseline or candidate with selected Nsight Compute metrics.")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--mode", choices=["ncu-selected"], default="ncu-selected")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--candidate-source", help="Optional candidate source containing ModelNew.")
    parser.add_argument("--metrics-config", "--ncu-metrics", dest="metrics_config", default="configs/ncu_metrics.yaml")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output", help="Output JSONL path.")
    args = parser.parse_args()

    stamp = utc_timestamp().replace(":", "").replace("+", "Z")
    backend = "candidate" if args.candidate_source else "pytorch_eager"
    prefix = os.environ.get("PROFBRIDGE_EXPERIMENT_PREFIX", "ncu")
    experiment_id = f"{prefix}_l{args.level}_t{args.task}_{backend}_{int(time.time())}"
    out_path = Path(args.output) if args.output else ensure_dir("results/profiles") / f"profile_{experiment_id}.jsonl"
    baseline_out = ensure_dir("results/profiles") / f"profile_target_{experiment_id}.jsonl"
    command = [
        sys.executable,
        "scripts/run_kernelbench_baseline.py",
        "--level",
        str(args.level),
        "--task",
        str(args.task),
        "--backend",
        backend,
        "--num-warmup",
        str(args.num_warmup),
        "--num-repeats",
        str(args.num_repeats),
        "--device",
        "0",
        "--output",
        str(baseline_out),
    ]
    if args.candidate_source:
        command.extend(["--candidate-source", args.candidate_source])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    profile = run_ncu_profile(
        command,
        requested_metrics=load_requested_metrics(args.metrics_config),
        output_dir="results/profiles",
        experiment_id=experiment_id,
        timeout=args.timeout,
        env=env,
    )
    record = {
        "experiment_id": experiment_id,
        "level": args.level,
        "task_id": args.task,
        "backend": backend,
        "candidate_source_path": args.candidate_source,
        "mode": args.mode,
        "profile": profile,
        "environment": collect_environment_metadata(args.device, REPO_ROOT),
        "timestamp": utc_timestamp(),
    }
    append_jsonl(out_path, record)

    report = Path("reports/ncu_smoke.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# NCU Smoke",
                "",
                f"- Experiment: `{experiment_id}`",
                f"- Status: `{profile.get('profiling_status')}`",
                f"- Failure reason: `{profile.get('failure_reason')}`",
                f"- Metrics requested: {len(profile.get('metrics_requested', []))}",
                f"- Metrics available: {len(profile.get('metrics_available', []))}",
                f"- Metrics unavailable: {len(profile.get('metrics_unavailable', []))}",
                f"- Raw output: `{profile.get('raw_output_path')}`",
                f"- Result JSONL: `{out_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(out_path)
    print(report)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if profile.get("profiling_status") in {"success", "partial_success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
