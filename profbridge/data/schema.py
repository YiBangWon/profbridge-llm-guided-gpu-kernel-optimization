from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROFILE_PAIR_REQUIRED = [
    "experiment_id",
    "task_id",
    "level",
    "candidate_id",
    "parent_candidate_id",
    "candidate_source_path",
    "source_hash",
    "generation_method",
    "transformation_history",
    "correctness_pass",
    "correctness_error",
    "latency_stats",
    "cheap_features",
    "high_fidelity_profile",
    "predictor",
    "cost_accounting",
    "environment",
]

BASELINE_REQUIRED = [
    "experiment_id",
    "task_id",
    "level",
    "backend",
    "correctness_pass",
    "correctness_error",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_std_ms",
    "latency_p05_ms",
    "latency_p95_ms",
    "num_warmup",
    "num_repeats",
    "raw_latency_file",
    "gpu_name",
    "driver_version",
    "cuda_version",
    "torch_version",
    "python_version",
    "git_commit",
    "timestamp",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_required(record: dict[str, Any], required: list[str]) -> list[str]:
    return [key for key in required if key not in record]


def validate_profile_pair(record: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    missing = validate_required(record, PROFILE_PAIR_REQUIRED)
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    if record.get("generation_method") not in {
        "pytorch_baseline",
        "torch_compile",
        "llm_plan_impl",
        "mutation",
        "manual",
        "mock_llm",
    }:
        errors.append(f"invalid generation_method: {record.get('generation_method')}")
    for nested in ["latency_stats", "cheap_features", "high_fidelity_profile", "predictor", "cost_accounting", "environment"]:
        if nested in record and not isinstance(record[nested], dict):
            errors.append(f"{nested} must be an object")
    return not errors, errors


def validate_baseline(record: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = validate_required(record, BASELINE_REQUIRED)
    return not missing, [f"missing required keys: {', '.join(missing)}"] if missing else []


def example_profile_pair() -> dict[str, Any]:
    return {
        "experiment_id": "example",
        "task_id": 1,
        "level": 1,
        "candidate_id": "l1_t1_pytorch_eager",
        "parent_candidate_id": None,
        "candidate_source_path": "external/KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py",
        "source_hash": "example-source-hash",
        "generation_method": "pytorch_baseline",
        "transformation_history": [],
        "optimization_plan_text": None,
        "prompt_hash": None,
        "model_name": None,
        "correctness_pass": True,
        "correctness_error": None,
        "latency_stats": {
            "mean_ms": 1.0,
            "median_ms": 1.0,
            "std_ms": 0.0,
            "p05_ms": 1.0,
            "p95_ms": 1.0,
            "num_warmup": 10,
            "num_repeats": 50,
            "raw_latency_file": "results/baselines/example_raw.jsonl",
        },
        "cheap_features": {
            "source_length": 512,
            "source_hash": "example-source-hash",
            "num_cuda_kernels": 0,
            "feature_errors": {},
        },
        "high_fidelity_profile": {
            "selected_ncu_metrics": {"gpu__time_duration.sum": 1.0},
            "unavailable_metrics": [],
            "profiling_time_sec": 0.5,
            "profiling_status": "ok",
        },
        "predictor": {
            "predicted_metrics": {},
            "uncertainty": None,
            "used_full_profile": True,
        },
        "cost_accounting": {
            "llm_time_sec": 0.0,
            "compile_time_sec": 0.0,
            "timing_time_sec": 0.1,
            "static_feature_time_sec": 0.01,
            "ncu_profile_time_sec": 0.5,
            "predictor_inference_time_sec": 0.0,
            "predictor_update_time_sec": 0.0,
            "total_wall_clock_sec": 0.61,
        },
        "environment": {
            "gpu_name": "example",
            "driver_version": "example",
            "cuda_version": "example",
            "ncu_version": "example",
            "torch_version": "example",
            "python_version": "3.10",
            "git_commit": "example",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    }


def write_example(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(example_profile_pair(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
