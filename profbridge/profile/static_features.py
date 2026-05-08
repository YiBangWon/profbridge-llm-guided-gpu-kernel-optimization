from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import Any

from profbridge.data.schema import sha256_file


CUDA_KERNEL_PATTERNS = [
    r"__global__",
    r"@triton\.jit",
    r"load_inline",
    r"CUDAExtension",
]


def _count_regex(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


class _LoopCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loops = 0

    def visit_For(self, node):  # noqa: N802
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node):  # noqa: N802
        self.loops += 1
        self.generic_visit(node)


def extract_static_features(
    candidate_path: str | Path,
    *,
    operator_shape_features: dict[str, Any] | None = None,
    transformation_history: list[str] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    path = Path(candidate_path)
    errors: dict[str, str] = {}
    text = ""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
        errors["source_decode"] = "decoded with replacement"
    except Exception as exc:
        errors["source_read"] = f"{type(exc).__name__}: {exc}"

    loop_count = None
    if text and path.suffix == ".py":
        try:
            tree = ast.parse(text)
            counter = _LoopCounter()
            counter.visit(tree)
            loop_count = counter.loops
        except Exception as exc:
            errors["python_ast"] = f"{type(exc).__name__}: {exc}"

    features: dict[str, Any] = {
        "candidate_path": str(path),
        "source_hash": sha256_file(path) if path.exists() else None,
        "source_length": len(text),
        "num_lines": text.count("\n") + (1 if text else 0),
        "num_cuda_kernels": sum(_count_regex(pattern, text) for pattern in CUDA_KERNEL_PATTERNS),
        "approx_num_loops": loop_count if loop_count is not None else _count_regex(r"\b(for|while)\b", text),
        "uses_shared_memory_keyword": "__shared__" in text or "shared_memory" in text or "tl.store" in text,
        "uses_atomic": "atomic" in text.lower(),
        "uses_syncthreads": "__syncthreads" in text,
        "uses_tensor_core_keywords": any(key in text.lower() for key in ["wmma", "mma.", "tensorcore", "cublas", "cudnn"]),
        "uses_triton": "triton" in text.lower() or "@triton.jit" in text,
        "uses_torch_compile": "torch.compile" in text,
        "operator_shape_features": operator_shape_features or {},
        "transformation_history": transformation_history or [],
        "feature_errors": errors,
        "static_feature_time_sec": time.perf_counter() - start,
    }
    if path.exists():
        try:
            features["source_file_size_bytes"] = path.stat().st_size
        except Exception as exc:
            errors["stat"] = f"{type(exc).__name__}: {exc}"
    return features
