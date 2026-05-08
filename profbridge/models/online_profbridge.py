from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


def _num_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[str(k)] = float(v)
        elif isinstance(v, dict) and isinstance(v.get("mean"), (int, float)) and math.isfinite(float(v["mean"])):
            out[str(k)] = float(v["mean"])
    return out


def _key(features: dict[str, Any], *names: str) -> tuple[Any, ...]:
    return tuple(features.get(n) for n in names)


@dataclass
class OnlineProfBridge:
    """Small online numerical profiler predictor for replay/live smoke.

    The model avoids raw identity-hash categorical leakage.  It predicts each
    target from hierarchical means: global, source, task, and task+source.  The
    uncertainty combines ensemble disagreement, rolling residuals, low-count
    penalties, and a light OOD penalty.  This is intentionally simple and
    online-friendly for P07 smoke, not the final predictor architecture.
    """

    residual_window: int = 64
    min_group_count: int = 3
    target_names: set[str] = field(default_factory=set)
    global_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    global_count: Counter[str] = field(default_factory=Counter)
    group_sum: dict[str, dict[tuple[Any, ...], dict[str, float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    group_count: dict[str, dict[tuple[Any, ...], Counter[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(Counter)))
    residuals: dict[str, deque[float]] = field(default_factory=dict)
    feature_reservoir: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=256))
    seen: int = 0

    def reset(self) -> None:
        self.target_names.clear()
        self.global_sum.clear(); self.global_count.clear()
        self.group_sum.clear(); self.group_count.clear()
        self.residuals.clear(); self.feature_reservoir.clear()
        self.seen = 0

    def fit_initial(self, records: list[dict[str, Any]]) -> dict[str, float]:
        start = time.perf_counter()
        self.reset()
        for record in records:
            self.update(record.get("cheap_features_for_replay") or record, record.get("measured_metrics") or {})
        return {"fit_initial_time_sec": time.perf_counter() - start, "initial_records": len(records), "target_count": len(self.target_names)}

    def predict(self, cheap_features: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        start = time.perf_counter()
        if not self.target_names:
            return {}, {"global": float("inf"), "inference_time_sec": time.perf_counter() - start, "reason": "no_targets"}
        preds: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        flat = self._numeric_feature_view(cheap_features)
        ood = self._ood_distance(flat)
        for target in sorted(self.target_names):
            values: list[float] = []
            counts: list[int] = []
            base = self._global_mean(target)
            if base is not None:
                values.append(base); counts.append(self.global_count[target])
            for group_name, key in self._group_keys(cheap_features):
                mean, count = self._group_mean(group_name, key, target)
                if mean is not None:
                    values.append(mean); counts.append(count)
            if not values:
                preds[target] = 0.0
                uncertainty[target] = float("inf")
                continue
            # Prefer the most specific populated group, while retaining ensemble
            # disagreement as uncertainty.
            pred = values[-1]
            disagreement = self._normalized_std(values)
            residual = self._rolling_residual(target)
            low_count = 1.0 / math.sqrt(max(1, counts[-1]))
            uncertainty[target] = disagreement + residual + low_count + 0.05 * ood
            preds[target] = pred
        uncertainty["global"] = max(v for k, v in uncertainty.items() if k != "global") if uncertainty else float("inf")
        uncertainty["ood_distance"] = ood
        uncertainty["inference_time_sec"] = time.perf_counter() - start
        return preds, uncertainty

    def update(self, cheap_features: dict[str, Any], measured_metrics: dict[str, Any]) -> dict[str, float]:
        start = time.perf_counter()
        targets = _num_metrics(measured_metrics)
        if not targets:
            return {"predictor_update_time_sec": time.perf_counter() - start, "num_targets": 0}
        prior, _ = self.predict(cheap_features) if self.target_names else ({}, {})
        for target, y in targets.items():
            if target in prior and math.isfinite(prior[target]):
                denom = max(abs(y), 1.0)
                self.residuals.setdefault(target, deque(maxlen=self.residual_window)).append(abs(prior[target] - y) / denom)
            self.global_sum[target] += y
            self.global_count[target] += 1
            for group_name, key in self._group_keys(cheap_features):
                self.group_sum[group_name][key][target] += y
                self.group_count[group_name][key][target] += 1
        self.target_names.update(targets)
        self.feature_reservoir.append(self._numeric_feature_view(cheap_features))
        self.seen += 1
        return {"predictor_update_time_sec": time.perf_counter() - start, "num_targets": len(targets)}

    def should_profile(self, uncertainty: dict[str, float], threshold: float | None = None, policy_config: dict[str, Any] | None = None) -> bool:
        policy_config = policy_config or {}
        policy = policy_config.get("policy", policy_config.get("mode", "online_profbridge"))
        step = int(policy_config.get("step", 0))
        threshold = float(policy_config.get("threshold", threshold if threshold is not None else 1.0))
        if policy == "always_profile":
            return True
        if policy in {"never_profile", "cheap_only", "no_profile"}:
            return False
        if policy == "random_profile_budget":
            rate = float(policy_config.get("profile_probability", policy_config.get("budget_fraction", 0.1)))
            return random.Random(int(policy_config.get("seed", 0)) + step).random() < rate
        if policy == "periodic_profile":
            period = max(1, int(policy_config.get("period", 5)))
            return step % period == 0
        if policy == "hybrid_periodic_uncertainty":
            first_k = max(0, int(policy_config.get("first_k", 2)))
            every_n = max(1, int(policy_config.get("every_n", 4)))
            profile_count = int(policy_config.get("profile_count", 0))
            seen_count = max(1, int(policy_config.get("seen_count", step + 1)))
            max_profile_fraction = float(policy_config.get("max_profile_fraction", 0.5))
            if step < first_k:
                return True
            if profile_count / seen_count >= max_profile_fraction:
                return False
            if (step - first_k) % every_n == 0:
                return True
            return float(uncertainty.get("global", float("inf"))) > threshold
        if policy in {"uncertainty_threshold", "online_profbridge", "static_profbridge"}:
            return float(uncertainty.get("global", float("inf"))) > threshold
        return True

    def export_state(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "target_count": len(self.target_names),
            "targets": sorted(self.target_names),
            "global_count": dict(self.global_count),
            "residual_targets": {k: len(v) for k, v in self.residuals.items()},
        }

    def _group_keys(self, features: dict[str, Any]) -> list[tuple[str, tuple[Any, ...]]]:
        return [
            ("source", _key(features, "candidate_source_type")),
            ("task", _key(features, "task_id")),
            ("task_source", _key(features, "task_id", "candidate_source_type")),
            ("task_generation", _key(features, "task_id", "generation_method")),
        ]

    def _global_mean(self, target: str) -> float | None:
        c = self.global_count[target]
        return self.global_sum[target] / c if c else None

    def _group_mean(self, group_name: str, key: tuple[Any, ...], target: str) -> tuple[float | None, int]:
        c = self.group_count[group_name][key][target]
        if not c:
            return None, 0
        return self.group_sum[group_name][key][target] / c, c

    def _rolling_residual(self, target: str) -> float:
        vals = self.residuals.get(target)
        if not vals:
            return 0.5
        return sum(vals) / len(vals)

    def _normalized_std(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.5
        mean = sum(values) / len(values)
        denom = max(abs(mean), 1.0)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) / denom

    def _numeric_feature_view(self, features: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        lat = features.get("latency_stats") if isinstance(features.get("latency_stats"), dict) else {}
        for key in ["mean_ms", "median_ms", "std_ms", "p05_ms", "p95_ms"]:
            val = lat.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                out[f"latency.{key}"] = float(val)
        cheap = features.get("cheap_features") if isinstance(features.get("cheap_features"), dict) else {}
        for key in ["source_length", "source_file_size_bytes", "num_lines", "approx_num_loops", "num_cuda_kernels"]:
            val = cheap.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                out[f"cheap.{key}"] = float(val)
        return out

    def _ood_distance(self, features: dict[str, float]) -> float:
        if not self.feature_reservoir or not features:
            return 1.0
        dists = []
        for prev in self.feature_reservoir:
            keys = set(features) | set(prev)
            if not keys:
                continue
            dists.append(math.sqrt(sum((features.get(k, 0.0) - prev.get(k, 0.0)) ** 2 for k in keys) / len(keys)))
        return min(dists) if dists else 1.0
