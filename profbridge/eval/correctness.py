from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CorrectnessResult:
    passed: bool
    error: str | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None


def move_to_device(value: Any, device: str):
    try:
        import torch  # type: ignore
    except Exception:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def detach_to_cpu(value: Any):
    try:
        import torch  # type: ignore
    except Exception:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    return value


def _compare_tensors(actual, expected, rtol: float, atol: float) -> CorrectnessResult:
    import torch  # type: ignore

    if actual.shape != expected.shape:
        return CorrectnessResult(False, f"shape mismatch: actual {tuple(actual.shape)} expected {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        actual = actual.to(expected.dtype)
    diff = (actual - expected).detach()
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    denom = expected.detach().abs().clamp_min(atol)
    max_rel = float((diff.abs() / denom).max().item()) if diff.numel() else 0.0
    ok = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
    return CorrectnessResult(ok, None if ok else f"allclose failed max_abs={max_abs:.6g} max_rel={max_rel:.6g}", max_abs, max_rel)


def compare_outputs(actual: Any, expected: Any, rtol: float = 1e-3, atol: float = 1e-3) -> CorrectnessResult:
    import torch  # type: ignore

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return _compare_tensors(actual, expected, rtol, atol)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return CorrectnessResult(False, f"sequence length mismatch: actual {len(actual)} expected {len(expected)}")
        worst_abs = 0.0
        worst_rel = 0.0
        for idx, (act_item, exp_item) in enumerate(zip(actual, expected)):
            result = compare_outputs(act_item, exp_item, rtol, atol)
            worst_abs = max(worst_abs, result.max_abs_error or 0.0)
            worst_rel = max(worst_rel, result.max_rel_error or 0.0)
            if not result.passed:
                result.error = f"output[{idx}]: {result.error}"
                return result
        return CorrectnessResult(True, None, worst_abs, worst_rel)
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return CorrectnessResult(False, "dict key mismatch")
        for key in expected:
            result = compare_outputs(actual[key], expected[key], rtol, atol)
            if not result.passed:
                result.error = f"output[{key!r}]: {result.error}"
                return result
        return CorrectnessResult(True)
    return CorrectnessResult(actual == expected, None if actual == expected else "non-tensor output mismatch")


def run_correctness(reference_model: Any, candidate_model: Any, inputs: list[Any], device: str, rtol: float = 1e-3, atol: float = 1e-3) -> CorrectnessResult:
    try:
        import torch  # type: ignore

        reference_model = reference_model.to(device).eval()
        candidate_model = candidate_model.to(device).eval()
        device_inputs = move_to_device(inputs, device)
        with torch.no_grad():
            expected = reference_model(*device_inputs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                expected = detach_to_cpu(expected)
                torch.cuda.empty_cache()
            actual = candidate_model(*device_inputs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                actual = detach_to_cpu(actual)
                torch.cuda.empty_cache()
        return compare_outputs(actual, expected, rtol=rtol, atol=atol)
    except Exception as exc:
        return CorrectnessResult(False, f"{type(exc).__name__}: {exc}")
