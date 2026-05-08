from __future__ import annotations

import hashlib
import math
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def score_value_of_profile(sketch: dict[str, Any]) -> float:
    acq = sketch.get("acquisition") or {}
    return fnum(acq.get("value_of_profile"))


def _uncertainty(sketch: dict[str, Any]) -> float:
    return fnum((sketch.get("acquisition") or {}).get("uncertainty_score"))


def _benefit(sketch: dict[str, Any]) -> float:
    return fnum((sketch.get("acquisition") or {}).get("predicted_benefit"))


def _cost(sketch: dict[str, Any]) -> float:
    return max(fnum((sketch.get("acquisition") or {}).get("profile_cost_estimate"), 8.0), 1e-6)


def _oracle_value(sketch: dict[str, Any]) -> float:
    cheap = sketch.get("cheap_features") or {}
    actual = fnum(cheap.get("actual_benefit_shadow"))
    return actual / _cost(sketch)


def _stable_random(sketch: dict[str, Any]) -> float:
    key = f"{sketch.get('task_id')}::{sketch.get('candidate_id')}".encode()
    return int(hashlib.sha256(key).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)


def rank_score(sketch: dict[str, Any], policy: str) -> float:
    if policy == "uncertainty_only":
        return _uncertainty(sketch)
    if policy == "predicted_benefit_only":
        return _benefit(sketch)
    if policy == "cost_only":
        return -_cost(sketch)
    if policy == "random_profile":
        return _stable_random(sketch)
    if policy == "oracle_vop":
        return _oracle_value(sketch)
    if policy == "value_of_profile":
        return score_value_of_profile(sketch)
    if policy == "periodic_profile":
        return 0.0
    return 0.0


def _geomean(values: list[float]) -> float | None:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def evaluate_policies(sketches: list[dict[str, Any]], budgets: list[float], policies: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(sketches, key=lambda s: (int(s.get("task_id") or 0), str(s.get("candidate_id") or "")))
    summaries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    n = len(ordered)
    always_count = n
    for budget in budgets:
        budget_count = max(0, min(n, int(round(n * budget))))
        for policy in policies:
            if policy == "always_profile":
                selected = set(range(n))
            elif policy == "cheap_only":
                selected = set()
            elif policy == "periodic_profile":
                if budget_count <= 0:
                    selected = set()
                else:
                    interval = max(1, round(n / budget_count))
                    selected = {i for i in range(n) if i % interval == 0}
                    selected = set(sorted(selected)[:budget_count])
            else:
                ranked = sorted(range(n), key=lambda i: rank_score(ordered[i], policy), reverse=True)
                selected = set(ranked[:budget_count])
            full_calls = len(selected)
            errors: list[float] = []
            unprofiled_errors: list[float] = []
            speedups_all: list[float] = []
            speedups_profiled: list[float] = []
            method_wall = 0.0
            shadow_wall = 0.0
            predictor_overhead = 0.0
            for i, sketch in enumerate(ordered):
                acq = sketch.get("acquisition") or {}
                cheap = sketch.get("cheap_features") or {}
                profile_cost = _cost(sketch)
                pred_error = fnum(acq.get("prediction_error_shadow"))
                profiled = i in selected
                err = 0.0 if profiled else pred_error
                errors.append(err)
                if not profiled:
                    unprofiled_errors.append(pred_error)
                    shadow_wall += profile_cost
                else:
                    method_wall += profile_cost
                predictor_overhead += 0.003
                sp = fnum(cheap.get("speedup_over_eager"), 0.0)
                if sp > 0:
                    speedups_all.append(sp)
                    if profiled:
                        speedups_profiled.append(sp)
                decisions.append({
                    "budget_fraction": budget,
                    "policy_name": policy,
                    "stream_index": i,
                    "task_id": sketch.get("task_id"),
                    "candidate_id": sketch.get("candidate_id"),
                    "used_full_profile": profiled,
                    "profile_decision": "full_profile" if profiled else "predict_only",
                    "profile_reason": policy,
                    "value_of_profile": acq.get("value_of_profile"),
                    "predicted_benefit": acq.get("predicted_benefit"),
                    "uncertainty_score": acq.get("uncertainty_score"),
                    "profile_cost_estimate": acq.get("profile_cost_estimate"),
                    "prediction_error_shadow": pred_error,
                    "feedback_error_contribution": err,
                })
            summaries.append({
                "budget_fraction": budget,
                "policy_name": policy,
                "candidate_count": n,
                "full_profile_call_count": full_calls,
                "full_profile_reduction_vs_always": 1.0 - (full_calls / always_count if always_count else 0.0),
                "feedback_error_shadow_mean": sum(errors) / len(errors) if errors else None,
                "unprofiled_prediction_error_shadow_mean": sum(unprofiled_errors) / len(unprofiled_errors) if unprofiled_errors else None,
                "method_wall_clock_sec_total": method_wall + predictor_overhead,
                "eval_only_shadow_profile_time_sec_total": shadow_wall,
                "predictor_overhead_sec_total": predictor_overhead,
                "best_speedup_all_timing_visible": max(speedups_all) if speedups_all else None,
                "best_speedup_profiled_subset": max(speedups_profiled) if speedups_profiled else None,
                "geomean_speedup_all_timing_visible": _geomean(speedups_all),
                "geomean_speedup_profiled_subset": _geomean(speedups_profiled),
            })
    return summaries, decisions
