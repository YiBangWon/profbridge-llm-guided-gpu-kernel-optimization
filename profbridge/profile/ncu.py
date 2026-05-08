from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from profbridge.utils.env import ensure_dir, load_config
from profbridge.utils.subprocess_utils import run_command


DEFAULT_METRICS = [
    "gpu__time_duration",
    "gpu__time_active",
    "sm__warps_active",
    "dram__bytes",
    "dram__bytes_read",
    "dram__bytes_write",
    "smsp__inst_executed",
    "smsp__cycles_active",
    "smsp__cycles_elapsed",
]


def load_requested_metrics(config_path: str | Path | None = None) -> list[str]:
    if config_path is None:
        return list(DEFAULT_METRICS)
    data = load_config(config_path)
    metrics = data.get("metrics", data if isinstance(data, list) else None)
    return [str(metric) for metric in (metrics or DEFAULT_METRICS)]


def resolve_ncu_bin() -> str | None:
    configured = os.environ.get("NCU_BIN")
    if configured:
        return configured if Path(configured).exists() else None
    return shutil.which("ncu")


def query_available_metrics(timeout: float = 30) -> tuple[set[str] | None, str | None]:
    ncu_bin = resolve_ncu_bin()
    if not ncu_bin:
        return set(), "ncu not found"
    attempts = [
        [ncu_bin, "--query-metrics"],
        [ncu_bin, "--query-metrics", "--csv"],
    ]
    for command in attempts:
        result = run_command(command, timeout=timeout)
        text = result.stdout + "\n" + result.stderr
        if result.ok and text.strip():
            names = set(re.findall(r"\b[a-zA-Z][\w$]*(?:__[\w$]+)+(?:\.[\w$%]+)*\b", text))
            if names:
                return names, None
    return None, "could not query metrics; ncu may still profile requested metrics"


def _to_float(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if cleaned in {"", "n/a", "N/A", "--"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _csv_from_first_header(text: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith('"ID",') or line.startswith("ID,"):
            return "\n".join(lines[idx:])
        if line.lower().startswith('"metric name",') or line.lower().startswith("metric name,"):
            return "\n".join(lines[idx:])
    return text


def _is_metric_column(name: str, requested_metrics: list[str] | None) -> bool:
    if requested_metrics:
        return any(name == metric or name.startswith(f"{metric}.") for metric in requested_metrics)
    return "__" in name and not name.startswith(("device__attribute_", "numa__", "nvlink__", "profiler__"))


def parse_ncu_csv(text: str, requested_metrics: list[str] | None = None) -> dict[str, Any]:
    parsed: dict[str, dict[str, Any]] = {}
    if not text.strip():
        return parsed
    try:
        reader = csv.DictReader(io.StringIO(_csv_from_first_header(text)))
        for row in reader:
            lower = {key.lower().strip(): value for key, value in row.items() if key is not None}
            metric = lower.get("metric name") or lower.get("metric") or lower.get("name")
            value = lower.get("metric value") or lower.get("value")
            unit = lower.get("metric unit") or lower.get("unit")
            if metric and value is not None:
                numeric = _to_float(value)
                if numeric is None:
                    continue
                bucket = parsed.setdefault(metric, {"values": [], "unit": unit})
                bucket["values"].append(numeric)
                if unit:
                    bucket["unit"] = unit
                continue
            if not reader.fieldnames:
                continue
            # Nsight Compute --csv --page raw emits one wide row per kernel, with
            # selected metrics as columns such as gpu__time_duration.sum.
            for field in reader.fieldnames:
                if not field or not _is_metric_column(field, requested_metrics):
                    continue
                numeric = _to_float(row.get(field, ""))
                if numeric is None:
                    continue
                parsed.setdefault(field, {"values": [], "unit": None})["values"].append(numeric)
    except Exception:
        return {}
    for metric, bucket in parsed.items():
        values = bucket.get("values", [])
        bucket["mean"] = sum(values) / len(values) if values else None
        bucket["count"] = len(values)
    return parsed


def run_ncu_profile(
    command: Sequence[str],
    *,
    requested_metrics: list[str],
    output_dir: str | Path,
    experiment_id: str,
    timeout: float = 300,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    out_dir = ensure_dir(output_dir)
    raw_output_path = out_dir / f"{experiment_id}_ncu_raw.csv"
    result_json_path = out_dir / f"{experiment_id}_ncu.json"

    record: dict[str, Any] = {
        "profiling_status": "not_run",
        "ncu_bin": resolve_ncu_bin(),
        "profiling_time_sec": None,
        "metrics_requested": requested_metrics,
        "metrics_available": [],
        "metrics_unavailable": [],
        "parsed_metrics": {},
        "raw_output_path": str(raw_output_path),
        "failure_reason": None,
    }

    ncu_bin = resolve_ncu_bin()
    if not ncu_bin:
        record.update(
            {
                "profiling_status": "failed",
                "profiling_time_sec": time.perf_counter() - start,
                "metrics_unavailable": requested_metrics,
                "failure_reason": "ncu not found; set NCU_BIN if Nsight Compute is installed outside PATH",
            }
        )
        result_json_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record

    available, query_error = query_available_metrics()
    if available is None:
        metrics_to_run = requested_metrics
        record["metric_query_warning"] = query_error
    else:
        metrics_to_run = [metric for metric in requested_metrics if metric in available]
        record["metrics_unavailable"] = [metric for metric in requested_metrics if metric not in available]
    record["metrics_available"] = metrics_to_run
    if not metrics_to_run:
        record.update(
            {
                "profiling_status": "failed",
                "profiling_time_sec": time.perf_counter() - start,
                "failure_reason": "no requested metrics are available",
            }
        )
        result_json_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record

    ncu_cmd = [
        ncu_bin,
        "--target-processes",
        "all",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metrics_to_run),
        *list(command),
    ]
    completed = run_command(ncu_cmd, timeout=timeout, env=env)
    raw_text = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    raw_output_path.write_text(raw_text, encoding="utf-8", errors="replace")
    parsed = parse_ncu_csv(raw_text, requested_metrics=metrics_to_run)
    parsed_count = len(parsed)
    if parsed_count and completed.ok and not record.get("metrics_unavailable"):
        status = "success"
    elif parsed_count:
        status = "partial_success"
    else:
        status = "failed"
    record.update(
        {
            "profiling_status": status,
            "profiling_time_sec": time.perf_counter() - start,
            "parsed_metrics": parsed,
            "failure_reason": None if parsed_count else (completed.error or completed.stderr.strip() or "no parseable NCU metrics"),
            "returncode": completed.returncode,
        }
    )
    result_json_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record
