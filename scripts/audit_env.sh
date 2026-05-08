#!/usr/bin/env bash
set -u

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/audit_env.sh

Audits the local environment and writes:
  reports/env_audit_<timestamp>.md
  reports/env_audit_<timestamp>.json
EOF
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p reports logs

PYTHON_CMD=()
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("$PYTHON")
elif command -v python3 >/dev/null 2>&1 && python3 --version >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python.exe >/dev/null 2>&1 && python.exe --version >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v py >/dev/null 2>&1 && py -3 --version >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
else
  echo "No usable Python interpreter found" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" - <<'PY'
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, timeout=20):
    try:
        cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": cp.returncode,
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
            "available": cp.returncode == 0,
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "available": False}


def which(name):
    path = shutil.which(name)
    return {"available": path is not None, "path": path}


ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
reports = Path("reports")
json_path = reports / f"env_audit_{ts}.json"
md_path = reports / f"env_audit_{ts}.md"

audit = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cwd": str(Path.cwd()),
    "write_permission": os.access(Path.cwd(), os.W_OK),
    "os": {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    },
    "commands": {},
}

command_specs = {
    "kernel": ["uname", "-a"],
    "cpu": ["bash", "-lc", "lscpu || sysctl -a | grep machdep.cpu | head"],
    "memory": ["bash", "-lc", "free -h || vm_stat || true"],
    "disk": ["bash", "-lc", "df -h ."],
    "nvidia_smi": ["nvidia-smi"],
    "nvcc": ["nvcc", "--version"],
    "ncu": ["ncu", "--version"],
    "python": [sys.executable, "--version"],
    "conda": ["conda", "--version"],
    "mamba": ["mamba", "--version"],
    "git": ["git", "--version"],
    "gcc": ["gcc", "--version"],
    "g++": ["g++", "--version"],
    "cmake": ["cmake", "--version"],
    "ninja": ["ninja", "--version"],
}

for name, cmd in command_specs.items():
    audit["commands"][name] = run(cmd)

for name in ["python", "python3", "conda", "mamba", "git", "gcc", "g++", "cmake", "ninja", "nvcc", "ncu", "nvidia-smi"]:
    audit[f"which_{name}"] = which(name)

try:
    import torch
    torch_info = {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            torch_info["devices"].append({"index": idx, "name": torch.cuda.get_device_name(idx)})
    audit["pytorch"] = torch_info
except Exception as exc:
    audit["pytorch"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# ProfBridge Environment Audit",
    "",
    f"- Timestamp: `{audit['timestamp']}`",
    f"- CWD: `{audit['cwd']}`",
    f"- Write permission: `{audit['write_permission']}`",
    f"- Platform: `{audit['os']['platform']}`",
    "",
    "## Tool Availability",
    "",
    "| Tool | Available | Detail |",
    "|---|---:|---|",
]
for name in ["nvidia_smi", "nvcc", "ncu", "python", "conda", "mamba", "git", "gcc", "g++", "cmake", "ninja"]:
    item = audit["commands"].get(name, {})
    detail = (item.get("stdout") or item.get("stderr") or "").splitlines()
    lines.append(f"| {name} | {item.get('available')} | `{detail[0] if detail else ''}` |")
lines.extend([
    "",
    "## PyTorch",
    "",
    "```json",
    json.dumps(audit["pytorch"], indent=2, sort_keys=True),
    "```",
    "",
    "## GPU",
    "",
    "```text",
    audit["commands"].get("nvidia_smi", {}).get("stdout") or audit["commands"].get("nvidia_smi", {}).get("stderr") or "missing",
    "```",
])
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json_path)
print(md_path)
PY
