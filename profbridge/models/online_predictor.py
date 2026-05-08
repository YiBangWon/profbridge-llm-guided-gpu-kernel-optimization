from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def flatten_features(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if obj is None:
        return out
    if isinstance(obj, bool):
        out[prefix or "value"] = 1.0 if obj else 0.0
    elif isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        out[prefix or "value"] = float(obj)
    elif isinstance(obj, str):
        if obj:
            out[f"{prefix}={obj}" if prefix else f"value={obj}"] = 1.0
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_features(value, child))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj[:16]):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(flatten_features(value, child))
    return out


@dataclass
class RunningNormalizer:
    count: dict[str, int] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    m2: dict[str, float] = field(default_factory=dict)

    def update(self, features: dict[str, float]) -> None:
        for key, value in features.items():
            n = self.count.get(key, 0) + 1
            delta = value - self.mean.get(key, 0.0)
            mean = self.mean.get(key, 0.0) + delta / n
            delta2 = value - mean
            self.count[key] = n
            self.mean[key] = mean
            self.m2[key] = self.m2.get(key, 0.0) + delta * delta2

    def transform(self, features: dict[str, float]) -> dict[str, float]:
        transformed = {"bias": 1.0}
        for key, value in features.items():
            n = self.count.get(key, 0)
            if n > 1:
                var = self.m2.get(key, 0.0) / max(1, n - 1)
                scale = math.sqrt(var) if var > 1e-12 else 1.0
                transformed[key] = (value - self.mean.get(key, 0.0)) / scale
            else:
                transformed[key] = value
        return transformed


@dataclass
class OnlineLinearMember:
    lr: float
    l2: float = 1e-5
    weights: dict[str, dict[str, float]] = field(default_factory=dict)

    def predict(self, features: dict[str, float], target: str) -> float:
        target_weights = self.weights.get(target, {})
        return sum(target_weights.get(key, 0.0) * value for key, value in features.items())

    def update(self, features: dict[str, float], targets: dict[str, float]) -> None:
        for target, y in targets.items():
            if not math.isfinite(float(y)):
                continue
            target_weights = self.weights.setdefault(target, {})
            pred = self.predict(features, target)
            err = pred - float(y)
            for key, value in features.items():
                old = target_weights.get(key, 0.0)
                grad = err * value + self.l2 * old
                target_weights[key] = old - self.lr * grad


@dataclass
class OnlineProfBridge:
    members: list[OnlineLinearMember] = field(default_factory=lambda: [
        OnlineLinearMember(0.03),
        OnlineLinearMember(0.01),
        OnlineLinearMember(0.003),
    ])
    normalizer: RunningNormalizer = field(default_factory=RunningNormalizer)
    target_names: set[str] = field(default_factory=set)
    residuals: dict[str, deque[float]] = field(default_factory=dict)
    reservoir: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=128))
    seen: int = 0

    def _prepare(self, cheap_features: dict[str, Any], update_stats: bool = False) -> dict[str, float]:
        flat = flatten_features(cheap_features)
        if update_stats:
            self.normalizer.update(flat)
        return self.normalizer.transform(flat)

    def predict(self, cheap_features: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        start = time.perf_counter()
        x = self._prepare(cheap_features, update_stats=False)
        predictions: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        targets = sorted(self.target_names)
        if not targets:
            return {}, {"global": float("inf"), "inference_time_sec": time.perf_counter() - start}
        ood = self._ood_distance(x)
        for target in targets:
            member_preds = [member.predict(x, target) for member in self.members]
            mean = sum(member_preds) / len(member_preds)
            variance = sum((value - mean) ** 2 for value in member_preds) / len(member_preds)
            residual = self._rolling_residual(target)
            predictions[target] = mean
            uncertainty[target] = math.sqrt(variance) + residual + 0.05 * ood
        uncertainty["global"] = max(uncertainty.values()) if uncertainty else float("inf")
        uncertainty["ood_distance"] = ood
        uncertainty["inference_time_sec"] = time.perf_counter() - start
        return predictions, uncertainty

    def update(self, cheap_features: dict[str, Any], measured_ncu_metrics: dict[str, Any]) -> dict[str, float]:
        start = time.perf_counter()
        targets = _extract_numeric_targets(measured_ncu_metrics)
        x_before = self._prepare(cheap_features, update_stats=False)
        for target, y in targets.items():
            if self.target_names:
                pred = sum(member.predict(x_before, target) for member in self.members) / len(self.members)
                self.residuals.setdefault(target, deque(maxlen=64)).append(abs(pred - y))
        self.target_names.update(targets)
        x = self._prepare(cheap_features, update_stats=True)
        for member in self.members:
            member.update(x, targets)
        self.reservoir.append(x)
        self.seen += 1
        return {"predictor_update_time_sec": time.perf_counter() - start, "num_targets": len(targets)}

    def should_profile(self, uncertainty: dict[str, float], threshold: float, policy_config: dict[str, Any]) -> bool:
        return should_profile(uncertainty, threshold, policy_config)

    def _rolling_residual(self, target: str) -> float:
        values = self.residuals.get(target)
        if not values:
            return 1.0
        return sum(values) / len(values)

    def _ood_distance(self, features: dict[str, float]) -> float:
        if not self.reservoir:
            return 1.0
        distances = []
        keys = set(features)
        for prior in self.reservoir:
            all_keys = keys | set(prior)
            distances.append(math.sqrt(sum((features.get(key, 0.0) - prior.get(key, 0.0)) ** 2 for key in all_keys)))
        return min(distances) if distances else 1.0


def _extract_numeric_targets(metrics: dict[str, Any]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            targets[key] = float(value)
        elif isinstance(value, dict):
            mean = value.get("mean")
            if isinstance(mean, (int, float)) and math.isfinite(float(mean)):
                targets[key] = float(mean)
    return targets


def should_profile(uncertainty: dict[str, float], threshold: float, policy_config: dict[str, Any]) -> bool:
    policy = policy_config.get("policy", policy_config.get("mode", "online_profbridge"))
    step = int(policy_config.get("step", 0))
    if policy == "always_profile":
        return True
    if policy in {"cheap_only", "no_profile"}:
        return False
    if policy == "random_profile_budget":
        rate = float(policy_config.get("profile_probability", policy_config.get("budget_fraction", 0.1)))
        rng = random.Random(int(policy_config.get("seed", 0)) + step)
        return rng.random() < rate
    if policy == "periodic_profile":
        period = max(1, int(policy_config.get("period", 5)))
        return step % period == 0
    if policy in {"static_profbridge", "online_profbridge"}:
        return float(uncertainty.get("global", float("inf"))) > threshold
    return True


def toy_online_update_reduces_error() -> bool:
    model = OnlineProfBridge()
    stream = [({"x": float(i)}, {"target_metric": 2.0 * i}) for i in range(1, 30)]
    first_pred, _ = model.predict(stream[-1][0])
    first_err = abs(first_pred.get("target_metric", 0.0) - stream[-1][1]["target_metric"])
    for features, target in stream:
        model.update(features, target)
    pred, _ = model.predict(stream[-1][0])
    final_err = abs(pred["target_metric"] - stream[-1][1]["target_metric"])
    return final_err < first_err
