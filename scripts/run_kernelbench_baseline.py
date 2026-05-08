from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.bench.kernelbench_loader import load_candidate_model, load_task
from profbridge.data.jsonl import append_jsonl
from profbridge.eval.correctness import run_correctness
from profbridge.eval.timing import time_model_forward
from profbridge.utils.env import collect_environment_metadata, ensure_dir, utc_timestamp


def _make_output_path(level: int, task: int, backend: str) -> Path:
    stamp = utc_timestamp().replace(":", "").replace("+", "Z")
    return ensure_dir("results/baselines") / f"baseline_l{level}_t{task}_{backend}_{stamp}.jsonl"


def _write_raw_latencies(path: Path, latencies: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, value in enumerate(latencies):
            handle.write(json.dumps({"iteration": idx, "latency_ms": value}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KernelBench PyTorch eager, torch.compile, or candidate baseline timing.")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--backend", choices=["pytorch_eager", "torch_compile", "candidate"], default="pytorch_eager")
    parser.add_argument("--candidate-source", help="Candidate source containing ModelNew when backend=candidate.")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", help="Output JSONL file.")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else _make_output_path(args.level, args.task, args.backend)
    raw_path = ensure_dir("results/baselines") / f"raw_l{args.level}_t{args.task}_{args.backend}_{int(time.time())}.jsonl"
    prefix = os.environ.get("PROFBRIDGE_EXPERIMENT_PREFIX", "baseline")
    experiment_id = f"{prefix}_l{args.level}_t{args.task}_{args.backend}_{int(time.time())}"
    env = collect_environment_metadata(args.device, REPO_ROOT)
    record = {
        "experiment_id": experiment_id,
        "task_id": args.task,
        "level": args.level,
        "backend": args.backend,
        "correctness_pass": False,
        "correctness_error": None,
        "latency_mean_ms": None,
        "latency_median_ms": None,
        "latency_std_ms": None,
        "latency_p05_ms": None,
        "latency_p95_ms": None,
        "num_warmup": args.num_warmup,
        "num_repeats": args.num_repeats,
        "raw_latency_file": str(raw_path),
        "gpu_name": env.get("gpu_name"),
        "driver_version": env.get("driver_version"),
        "cuda_version": env.get("cuda_version"),
        "torch_version": env.get("torch_version"),
        "python_version": env.get("python_version"),
        "git_commit": env.get("git_commit"),
        "timestamp": env.get("timestamp"),
        "environment": env,
        "timing_time_sec": 0.0,
        "failure_stage": None,
    }

    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(args.device)
        task = load_task(args.level, args.task, REPO_ROOT)
        torch.manual_seed(0)
        reference_model = task.make_model()
        if args.backend == "pytorch_eager":
            torch.manual_seed(0)
            candidate_model = task.make_model()
        elif args.backend == "torch_compile":
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available")
            torch.manual_seed(0)
            candidate_model = torch.compile(task.make_model())
        else:
            if not args.candidate_source:
                raise RuntimeError("--candidate-source is required for backend=candidate")
            torch.manual_seed(0)
            candidate_model = load_candidate_model(args.candidate_source)

        inputs = task.get_inputs()
        correctness = run_correctness(reference_model, candidate_model, inputs, f"cuda:{args.device}")
        record["correctness_pass"] = correctness.passed
        record["correctness_error"] = correctness.error
        record["max_abs_error"] = correctness.max_abs_error
        record["max_rel_error"] = correctness.max_rel_error
        if not correctness.passed:
            record["failure_stage"] = "correctness"
        else:
            timing = time_model_forward(candidate_model, inputs, args.num_warmup, args.num_repeats, args.device)
            record["timing_time_sec"] = timing.timing_time_sec
            if timing.error:
                record["failure_stage"] = "timing"
                record["correctness_error"] = timing.error
            for key, value in timing.stats.items():
                if key == "mean_ms":
                    record["latency_mean_ms"] = value
                elif key == "median_ms":
                    record["latency_median_ms"] = value
                elif key == "std_ms":
                    record["latency_std_ms"] = value
                elif key == "p05_ms":
                    record["latency_p05_ms"] = value
                elif key == "p95_ms":
                    record["latency_p95_ms"] = value
            _write_raw_latencies(raw_path, timing.latencies_ms)
    except Exception as exc:
        record["correctness_pass"] = False
        record["correctness_error"] = f"{type(exc).__name__}: {exc}"
        record["failure_stage"] = record.get("failure_stage") or "setup"
        _write_raw_latencies(raw_path, [])

    append_jsonl(out_path, record)
    print(out_path)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["correctness_pass"] and record["latency_mean_ms"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
