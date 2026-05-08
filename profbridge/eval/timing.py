from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from .correctness import move_to_device


@dataclass
class TimingResult:
    latencies_ms: list[float]
    stats: dict[str, float | int | None]
    timing_time_sec: float
    error: str | None = None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_latencies_ms(latencies_ms: list[float]) -> dict[str, float | int | None]:
    if not latencies_ms:
        return {
            "mean_ms": None,
            "median_ms": None,
            "std_ms": None,
            "p05_ms": None,
            "p95_ms": None,
            "num_repeats": 0,
        }
    return {
        "mean_ms": float(statistics.fmean(latencies_ms)),
        "median_ms": float(statistics.median(latencies_ms)),
        "std_ms": float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
        "p05_ms": percentile(latencies_ms, 0.05),
        "p95_ms": percentile(latencies_ms, 0.95),
        "num_repeats": len(latencies_ms),
    }


def time_cuda_callable(fn: Callable[[], Any], num_warmup: int, num_repeats: int, device: int = 0) -> TimingResult:
    start_wall = time.perf_counter()
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        with torch.no_grad():
            for _ in range(num_warmup):
                fn()
            torch.cuda.synchronize(device)
            latencies = []
            for _ in range(num_repeats):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                fn()
                end_event.record()
                torch.cuda.synchronize(device)
                latencies.append(float(start_event.elapsed_time(end_event)))
        stats = summarize_latencies_ms(latencies)
        return TimingResult(latencies, stats, time.perf_counter() - start_wall)
    except Exception as exc:
        return TimingResult([], summarize_latencies_ms([]), time.perf_counter() - start_wall, f"{type(exc).__name__}: {exc}")


def time_model_forward(model: Any, inputs: list[Any], num_warmup: int, num_repeats: int, device: int = 0) -> TimingResult:
    import torch  # type: ignore

    device_str = f"cuda:{device}"
    model = model.to(device_str).eval()
    device_inputs = move_to_device(inputs, device_str)

    def forward():
        return model(*device_inputs)

    return time_cuda_callable(forward, num_warmup=num_warmup, num_repeats=num_repeats, device=device)
