from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .schema import AcquisitionSketch, BottleneckSketch, GuidanceSketch, MetricEstimate, ProfileSketch

SELECTED_METRIC_PREFIXES = (
    "dram__bytes",
    "dram__bytes_read",
    "dram__bytes_write",
    "gpu__time_duration",
    "gpu__time_active",
    "sm__warps_active",
    "smsp__inst_executed",
    "smsp__cycles_active",
    "smsp__cycles_elapsed",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _flatten_profile_metrics(profile: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    selected = profile.get("selected_ncu_metrics") or {}
    for key, value in selected.items():
        x = fnum(value)
        if x is not None:
            metrics[key] = x
    parsed = profile.get("parsed_metrics") or {}
    for key, obj in parsed.items():
        if isinstance(obj, dict):
            x = fnum(obj.get("mean"))
        else:
            x = fnum(obj)
        if x is not None:
            metrics[key] = x
    return {k: v for k, v in metrics.items() if k.startswith(SELECTED_METRIC_PREFIXES)}


def extract_measured_metrics(row: dict[str, Any]) -> dict[str, float]:
    profile = row.get("high_fidelity_profile") or row.get("linked_high_fidelity_profile") or {}
    return _flatten_profile_metrics(profile)


def extract_profile_cost(row: dict[str, Any]) -> float:
    for key in ("ncu_profile_wall_sec", "ncu_profile_time_sec"):
        x = fnum(row.get(key))
        if x is not None:
            return x
    profile = row.get("high_fidelity_profile") or row.get("linked_high_fidelity_profile") or {}
    for key in ("profiling_time_sec", "profile_time_sec"):
        x = fnum(profile.get(key))
        if x is not None:
            return x
    cost = row.get("cost_accounting") or {}
    x = fnum(cost.get("ncu_profile_time_sec"))
    return x if x is not None else 8.0


def latency_ms(row: dict[str, Any]) -> float | None:
    for key in ("candidate_latency", "latency_mean_ms"):
        x = fnum(row.get(key))
        if x is not None:
            return x
    stats = row.get("latency_stats") or {}
    return fnum(stats.get("mean_ms"))


def signal_quality(signal: str | None) -> float:
    return {
        "predicted": 0.85,
        "predicted_label_only": 0.72,
        "oracle": 0.95,
        "measured": 0.95,
        "noisy": 0.35,
        "wrong": 0.25,
        "random": 0.2,
        "none": 0.1,
    }.get(str(signal or "none"), 0.45)


def derive_bottleneck(row: dict[str, Any], measured: dict[str, float]) -> tuple[str | None, list[str], list[str]]:
    label = row.get("bottleneck_label") or row.get("predicted_bottleneck")
    evidence: list[str] = []
    secondary: list[str] = []
    if not label:
        dram = measured.get("dram__bytes.avg") or measured.get("dram__bytes.sum")
        inst = measured.get("smsp__inst_executed.avg") or measured.get("smsp__inst_executed.sum")
        gpu = measured.get("gpu__time_duration.avg")
        if dram and inst:
            label = "memory_bound" if dram > inst else "compute_bound"
            evidence.extend(["dram__bytes", "smsp__inst_executed"])
        elif gpu:
            label = "latency_bound"
            evidence.append("gpu__time_duration")
        else:
            label = "unknown"
    if "dram" in json.dumps(measured) and label != "memory_bound":
        secondary.append("memory_bound")
    return label, secondary, evidence


def predicted_metric_estimates(row: dict[str, Any], signal: str | None) -> list[MetricEstimate]:
    vec = row.get("predicted_metric_vector") or row.get("metric_vector") or {}
    if isinstance(vec, dict):
        metrics = vec.get("metrics") if isinstance(vec.get("metrics"), dict) else vec
    else:
        metrics = {}
    out: list[MetricEstimate] = []
    uncertainty = fnum(row.get("uncertainty") or row.get("predicted_uncertainty"))
    if uncertainty is None:
        uncertainty = 1.0 - min(max(signal_quality(signal), 0.0), 1.0)
    for key, value in metrics.items():
        out.append(MetricEstimate(metric_name=str(key), predicted_value=fnum(value), uncertainty=uncertainty, missing=fnum(value) is None))
    if not out:
        out.append(MetricEstimate(metric_name="profile_metric_vector", predicted_value=None, uncertainty=uncertainty, missing=True))
    return out


def actual_benefit(row: dict[str, Any]) -> float:
    vals = [fnum(row.get("speedup_over_parent")), fnum(row.get("speedup_over_eager")), fnum(row.get("speedup_over_static_control_for_same_task"))]
    return max([max((x or 0.0) - 1.0, 0.0) for x in vals] or [0.0])


def predicted_benefit(row: dict[str, Any], signal: str | None, estimates: Iterable[MetricEstimate]) -> float:
    quality = signal_quality(signal)
    metric_mass = 0.0
    count = 0
    for est in estimates:
        if est.predicted_value is not None:
            metric_mass += math.log1p(abs(est.predicted_value))
            count += 1
    metric_component = (metric_mass / count / 16.0) if count else 0.05
    family_bonus = 0.08 if row.get("action_family_followed") else 0.0
    return max(0.0, min(2.5, quality * metric_component + family_bonus))


def build_sketch(row: dict[str, Any], phase: str = "unknown") -> ProfileSketch | None:
    task = row.get("task_id")
    candidate_id = row.get("candidate_id") or Path(str(row.get("candidate_path") or "candidate")).stem
    if task is None or not candidate_id:
        return None
    try:
        task_id = int(task)
    except Exception:
        return None
    measured = extract_measured_metrics(row)
    signal = row.get("bottleneck_signal_type") or row.get("oracle_or_predicted_or_corrupted") or row.get("signal_source") or "measured"
    estimates = predicted_metric_estimates(row, signal)
    label, secondary, evidence = derive_bottleneck(row, measured)
    cost = extract_profile_cost(row)
    unc = fnum(row.get("uncertainty"))
    if unc is None:
        unc = max(0.05, 1.0 - signal_quality(signal))
    pbenefit = predicted_benefit(row, signal, estimates)
    abenefit = actual_benefit(row)
    vop = (unc * max(pbenefit, 0.001)) / max(cost, 1e-6)
    pred_error = abs(abenefit - pbenefit)
    lat = latency_ms(row)
    cheap = {
        "latency_mean_ms": lat,
        "candidate_source_type": row.get("candidate_source_type"),
        "generation_method": row.get("generation_method"),
        "variant_id": row.get("variant_id"),
        "variant_name": row.get("variant_name"),
        "model_name": row.get("model_name"),
        "correctness_pass": row.get("correctness_pass"),
        "safety_pass": row.get("safety_pass", row.get("safety_scan_pass")),
        "robust_correctness_pass": row.get("robust_correctness_pass"),
        "ncu_success": row.get("ncu_success") if "ncu_success" in row else ((row.get("profiling_status") in {"success", "partial_success"}) if row.get("profiling_status") else None),
        "speedup_over_eager": row.get("speedup_over_eager"),
        "speedup_over_parent": row.get("speedup_over_parent"),
        "speedup_over_static_control_for_same_task": row.get("speedup_over_static_control_for_same_task"),
        "actual_benefit_shadow": abenefit,
    }
    guidance = GuidanceSketch(
        recommended_action_family=row.get("selected_action_family") or row.get("recommended_action_family"),
        discouraged_action_family=list(row.get("discouraged_actions") or []),
        rationale=f"{signal} bottleneck signal mapped to compact ProfileSketch guidance.",
        prompt_safe_text=f"Bottleneck: {label}; recommended action: {row.get('selected_action_family')}; preserve semantics and avoid fallbacks.",
    )
    return ProfileSketch(
        task_id=task_id,
        candidate_id=str(candidate_id),
        parent_candidate_id=row.get("parent_candidate_id"),
        cheap_features=cheap,
        predicted_metrics=estimates,
        measured_metrics=measured,
        bottleneck=BottleneckSketch(primary_label=label, secondary_labels=secondary, confidence=signal_quality(signal), evidence_metrics=evidence or list(measured)[:6]),
        acquisition=AcquisitionSketch(predicted_benefit=pbenefit, uncertainty_score=unc, profile_cost_estimate=cost, value_of_profile=vop, decision=None, prediction_error_shadow=pred_error),
        guidance=guidance,
        provenance={
            "phase": phase,
            "signal_source": signal,
            "predictor_version": "p10f_offline_profile_sketch_v1",
            "ncu_schema_version": row.get("ncu_metric_profile_version") or (row.get("high_fidelity_profile") or {}).get("ncu_metric_profile_version") or "unknown",
            "timestamp": row.get("timestamp"),
            "candidate_source_path": row.get("candidate_path") or row.get("candidate_source_path"),
            "diagnostic_oracle": signal == "oracle",
        },
    )
