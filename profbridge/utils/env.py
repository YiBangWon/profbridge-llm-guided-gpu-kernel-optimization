from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from .subprocess_utils import run_command


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def get_git_commit(cwd: str | Path | None = None) -> str | None:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None, timeout=5)
    return result.stdout.strip() if result.ok else None


def detect_nvidia_smi() -> dict[str, Any]:
    if not command_exists("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    query = "name,driver_version,memory.total"
    result = run_command(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
        timeout=10,
    )
    if not result.ok:
        return {
            "available": False,
            "reason": result.stderr.strip() or result.error or "nvidia-smi failed",
            "returncode": result.returncode,
        }
    gpus = []
    for idx, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": idx, "name": parts[0], "driver_version": parts[1], "memory_total": parts[2]})
    return {"available": True, "gpus": gpus}


def detect_cuda_version() -> str | None:
    if command_exists("nvcc"):
        result = run_command(["nvcc", "--version"], timeout=10)
        if result.ok:
            for line in result.stdout.splitlines():
                if "release" in line:
                    return line.strip()
    try:
        import torch  # type: ignore

        return torch.version.cuda
    except Exception:
        return None


def detect_ncu_version() -> str | None:
    if not command_exists("ncu"):
        return None
    result = run_command(["ncu", "--version"], timeout=10)
    text = (result.stdout + "\n" + result.stderr).strip()
    return text if result.ok and text else None


def detect_torch() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        info: dict[str, Any] = {
            "available": True,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            info["devices"] = [
                {"index": i, "name": torch.cuda.get_device_name(i)}
                for i in range(torch.cuda.device_count())
            ]
        return info
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def collect_environment_metadata(device: int | None = None, repo_root: str | Path | None = None) -> dict[str, Any]:
    nvidia = detect_nvidia_smi()
    gpu_name = None
    driver_version = None
    if nvidia.get("available") and nvidia.get("gpus"):
        gpus = nvidia["gpus"]
        idx = device if device is not None and 0 <= device < len(gpus) else 0
        gpu_name = gpus[idx].get("name")
        driver_version = gpus[idx].get("driver_version")

    torch_info = detect_torch()
    return {
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "cuda_version": detect_cuda_version(),
        "ncu_version": detect_ncu_version(),
        "torch_version": torch_info.get("torch_version"),
        "python_version": sys.version.split()[0],
        "git_commit": get_git_commit(repo_root),
        "timestamp": utc_timestamp(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        return json.loads(text)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
