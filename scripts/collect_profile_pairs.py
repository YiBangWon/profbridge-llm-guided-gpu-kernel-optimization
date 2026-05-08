from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from profbridge.bench.kernelbench_loader import find_task_path
from profbridge.data.jsonl import append_jsonl, read_jsonl
from profbridge.data.schema import sha256_file
from profbridge.profile.ncu import load_requested_metrics, run_ncu_profile
from profbridge.profile.static_features import extract_static_features
from profbridge.utils.env import collect_environment_metadata, ensure_dir, load_config, utc_timestamp
from profbridge.utils.subprocess_utils import run_command


SUCCESS_STATUSES = {"success", "partial_success"}


TASK_RISK_NOTES = {
    9: "medium: tall-skinny matmul with very large square output; smoke allowed with timeout",
    10: "medium: batched/3D matmul; smoke allowed with timeout",
    50: "medium: Conv2D AlexNet-style first layer; optional smoke allowed",
    54: "high: large Conv3D output and profiling cost; generate/dry-run only for P04C",
    56: "high: large Conv2D asymmetric output and profiling cost; generate/dry-run only for P04C",
}

CANDIDATE_SELECTION_PRIORITIES = {
    "pytorch_eager_baseline": 0,
    "manual_seed_candidate": 1,
    "generated_candidate": 1,
    "torch_compile_baseline": 2,
    "existing_kernelbench_solution": 3,
    "future_llm_generated_candidate": 4,
}


def _task_risk_note(task_id: int) -> str | None:
    return TASK_RISK_NOTES.get(int(task_id))


def stable_hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateSpec:
    generation_method: str
    source_path: Path
    backend: str
    candidate_source_type: str
    candidate_id: str
    transformation_history: tuple[str, ...] = ()
    parent_candidate_id: str | None = None
    notes: str | None = None
    is_baseline_candidate: bool = False
    is_generated_candidate: bool = False
    is_manual_seed: bool = False
    prompt_hash: str | None = None
    response_hash: str | None = None
    model_name: str | None = None
    provider: str | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    usage: dict[str, Any] | None = None
    estimated_cost_usd: float | None = None
    safety_scan_pass: bool | None = None
    safety_scan_issues: tuple[str, ...] = ()
    backend_config: dict[str, Any] = field(default_factory=dict)

    @property
    def source_hash(self) -> str:
        return sha256_file(self.source_path)

    @property
    def backend_config_hash(self) -> str:
        return stable_hash_json({"backend": self.backend, **self.backend_config})

    @property
    def candidate_config_hash(self) -> str:
        return stable_hash_json(
            {
                "candidate_id": self.candidate_id,
                "candidate_source_type": self.candidate_source_type,
                "generation_method": self.generation_method,
                "backend": self.backend,
                "backend_config_hash": self.backend_config_hash,
                "source_hash": self.source_hash,
                "transformation_history": list(self.transformation_history),
            }
        )

    @property
    def resume_key(self) -> str:
        return self.candidate_config_hash

    def dry_run_record(self, task_id: int, level: int, duplicate_source_hash: bool, duplicate_candidate_config: bool, run_ncu: bool, already_existing: bool = False, skip_reason: str | None = None) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "level": level,
            "candidate_id": self.candidate_id,
            "candidate_source_type": self.candidate_source_type,
            "generation_method": self.generation_method,
            "backend": self.backend,
            "candidate_source_path": str(self.source_path),
            "source_hash": self.source_hash,
            "backend_config_hash": self.backend_config_hash,
            "candidate_config_hash": self.candidate_config_hash,
            "duplicate_source_hash_in_task": duplicate_source_hash,
            "duplicate_candidate_config_hash_in_task": duplicate_candidate_config,
            "correctness_required": True,
            "ncu_profile_required_if_correct": bool(run_ncu),
            "parent_candidate_id": self.parent_candidate_id,
            "transformation_history": list(self.transformation_history),
            "is_baseline_candidate": self.is_baseline_candidate,
            "is_generated_candidate": self.is_generated_candidate,
            "is_manual_seed": self.is_manual_seed,
            "notes": self.notes,
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "usage": self.usage,
            "estimated_cost_usd": self.estimated_cost_usd,
            "safety_scan_pass": self.safety_scan_pass,
            "safety_scan_issues": list(self.safety_scan_issues),
            "already_existing_candidate": already_existing,
            "skip_reason": skip_reason,
        }



def _record_unique_key(record: dict[str, Any]) -> str | None:
    fields = ["task_id", "candidate_source_type", "generation_method", "source_hash", "candidate_config_hash"]
    values = []
    for field in fields:
        value = record.get(field)
        if value in {None, ""}:
            return None
        values.append(str(value))
    return "|".join(values)


def _candidate_unique_key(task_id: int, candidate: CandidateSpec) -> str:
    return "|".join(
        [
            str(task_id),
            candidate.candidate_source_type,
            candidate.generation_method,
            candidate.source_hash,
            candidate.candidate_config_hash,
        ]
    )


def _ncu_metrics_config(config: dict[str, Any]) -> str:
    return str(config.get("ncu_metrics_config", "configs/ncu_metrics.yaml"))


def _ncu_metric_profile_version(config: dict[str, Any]) -> str:
    configured = config.get("ncu_metric_profile_version")
    if configured:
        return str(configured)
    stem = Path(_ncu_metrics_config(config)).stem
    if "phase05b" in stem or "phase_05b" in stem:
        return "phase05b_v2"
    return "default_v1"


def _load_existing_unique_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    manifest = Path(path)
    if not manifest.exists():
        return set()
    keys: set[str] = set()
    for record in read_jsonl(manifest):
        key = record.get("unique_key") or _record_unique_key(record)
        if key:
            keys.add(str(key))
    return keys


def _existing_resume_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        keys = set()
        for record in read_jsonl(path):
            key = record.get("candidate_config_hash") or record.get("candidate_key")
            if not key and record.get("source_hash"):
                key = f"{record.get('generation_method')}:{record.get('backend')}:{record.get('candidate_id')}:{record.get('source_hash')}"
            if key:
                keys.add(str(key))
        return keys
    except Exception:
        return set()


def _mean_metrics(parsed: dict) -> dict[str, float]:
    out = {}
    for key, value in parsed.items():
        if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
            out[key] = float(value["mean"])
    return out


def _candidate_dirs(config: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    value = config.get(key)
    if value is None:
        return defaults
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _glob_candidate_files(directories: list[str], level: int, task_id: int) -> list[Path]:
    patterns = [
        f"{task_id}_*.py",
        f"{task_id}-*.py",
        f"task_{task_id}_*.py",
        f"task{task_id}_*.py",
        f"l{level}_t{task_id}_*.py",
        f"level{level}_task{task_id}_*.py",
    ]
    found: dict[str, Path] = {}
    task_subdirs = [
        Path(f"level{level}") / f"task_{task_id}",
        Path(f"level{level}") / f"task{task_id}",
        Path(f"l{level}") / f"task_{task_id}",
        Path(f"task_{task_id}"),
        Path(f"task{task_id}"),
    ]
    for directory in directories:
        root = (REPO_ROOT / directory).resolve() if not Path(directory).is_absolute() else Path(directory)
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file() and path.suffix == ".py":
                    found[str(path.resolve())] = path
        for rel_subdir in task_subdirs:
            subdir = root / rel_subdir
            if subdir.exists():
                for path in subdir.rglob("*.py"):
                    if path.is_file():
                        found[str(path.resolve())] = path
    return [found[key] for key in sorted(found)]



def _candidate_source_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "notes": None,
        "transformation_history": None,
        "generation_method": None,
        "prompt_hash": None,
        "response_hash": None,
        "model_name": None,
        "provider": None,
        "temperature": None,
        "reasoning_effort": None,
        "usage": None,
        "estimated_cost_usd": None,
        "safety_scan_pass": None,
        "safety_scan_issues": None,
    }
    sidecar = path.with_suffix(".metadata.json")
    if sidecar.exists():
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                metadata.update({k: raw.get(k) for k in metadata if k in raw})
                llm_log = raw.get("llm_log")
                if isinstance(llm_log, dict):
                    metadata["model_name"] = metadata.get("model_name") or llm_log.get("model_name")
                    metadata["provider"] = metadata.get("provider") or llm_log.get("provider")
                    metadata["temperature"] = metadata.get("temperature") if metadata.get("temperature") is not None else llm_log.get("temperature")
                    metadata["reasoning_effort"] = metadata.get("reasoning_effort") or llm_log.get("reasoning_effort")
                    metadata["usage"] = metadata.get("usage") or llm_log.get("usage")
                    metadata["estimated_cost_usd"] = metadata.get("estimated_cost_usd") if metadata.get("estimated_cost_usd") is not None else llm_log.get("estimated_cost_usd")
        except Exception:
            pass
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return metadata
    wanted = {
        "PROFBRIDGE_CANDIDATE_NOTES": "notes",
        "PROFBRIDGE_TRANSFORMATION_HISTORY": "transformation_history",
        "PROFBRIDGE_GENERATION_METHOD": "generation_method",
        "PROFBRIDGE_PROMPT_HASH": "prompt_hash",
        "PROFBRIDGE_RESPONSE_HASH": "response_hash",
        "PROFBRIDGE_MODEL_NAME": "model_name",
        "PROFBRIDGE_PROVIDER": "provider",
        "PROFBRIDGE_TEMPERATURE": "temperature",
        "PROFBRIDGE_REASONING_EFFORT": "reasoning_effort",
        "PROFBRIDGE_SAFETY_SCAN_PASS": "safety_scan_pass",
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue
                if target.id == "PROFBRIDGE_TRANSFORMATION_HISTORY" and isinstance(value, tuple):
                    value = list(value)
                metadata[wanted[target.id]] = value
    return metadata

def _file_candidate_id(prefix: str, path: Path, backend_config: dict[str, Any]) -> str:
    source_short = sha256_file(path)[:12]
    config_short = stable_hash_json(backend_config)[:12]
    return f"{prefix}_{path.stem}_{source_short}_{config_short}"

def enumerate_candidates(level: int, task_id: int, config: dict[str, Any]) -> list[CandidateSpec]:
    task_path = find_task_path(level, task_id, REPO_ROOT)
    candidates = [
        CandidateSpec(
            generation_method="pytorch_baseline",
            source_path=task_path,
            backend="pytorch_eager",
            candidate_source_type="pytorch_eager_baseline",
            candidate_id=f"l{level}_t{task_id}_pytorch_eager",
            transformation_history=("kernelbench_reference_model",),
            is_baseline_candidate=True,
            backend_config={"mode": "pytorch_eager"},
            notes="KernelBench PyTorch eager reference baseline.",
        )
    ]

    if bool(config.get("include_torch_compile", True)):
        candidates.append(
            CandidateSpec(
                generation_method="torch_compile",
                source_path=task_path,
                backend="torch_compile",
                candidate_source_type="torch_compile_baseline",
                candidate_id=f"l{level}_t{task_id}_torch_compile",
                transformation_history=("torch.compile",),
                is_baseline_candidate=True,
                backend_config={"mode": "torch_compile", "compile_backend": "default", "source_is_reference_task": True},
                notes="torch.compile baseline over the same KernelBench task source; candidate_config_hash distinguishes it from eager.",
            )
        )

    def add_candidate_files(
        *,
        directories: list[str],
        candidate_source_type: str,
        generation_method: str,
        id_label: str,
        default_history: tuple[str, ...],
        default_notes: str,
        is_generated_candidate: bool = False,
        is_manual_seed: bool = False,
    ) -> None:
        for idx, path in enumerate(_glob_candidate_files(directories, level, task_id)):
            backend_config = {"mode": "candidate_file", "source_registry": candidate_source_type}
            metadata = _candidate_source_metadata(path)
            if candidate_source_type == "future_llm_generated_candidate" and metadata.get("safety_scan_pass") is not True:
                continue
            candidate_notes = metadata.get("notes") if isinstance(metadata.get("notes"), str) else default_notes
            metadata_history = metadata.get("transformation_history")
            if isinstance(metadata_history, list) and all(isinstance(item, str) for item in metadata_history):
                history = tuple(metadata_history)
            else:
                history = default_history
            metadata_generation = metadata.get("generation_method")
            method = metadata_generation if isinstance(metadata_generation, str) else generation_method
            safety_issues_raw = metadata.get("safety_scan_issues")
            safety_issues = tuple(str(x) for x in safety_issues_raw) if isinstance(safety_issues_raw, list) else ()
            candidates.append(
                CandidateSpec(
                    generation_method=method,
                    source_path=path,
                    backend="candidate",
                    candidate_source_type=candidate_source_type,
                    candidate_id=_file_candidate_id(f"l{level}_t{task_id}_{id_label}_{idx}", path, backend_config),
                    transformation_history=history,
                    is_generated_candidate=is_generated_candidate,
                    is_manual_seed=is_manual_seed,
                    prompt_hash=metadata.get("prompt_hash") if isinstance(metadata.get("prompt_hash"), str) else None,
                    response_hash=metadata.get("response_hash") if isinstance(metadata.get("response_hash"), str) else None,
                    model_name=metadata.get("model_name") if isinstance(metadata.get("model_name"), str) else None,
                    provider=metadata.get("provider") if isinstance(metadata.get("provider"), str) else None,
                    temperature=float(metadata.get("temperature")) if isinstance(metadata.get("temperature"), (int, float)) else None,
                    reasoning_effort=metadata.get("reasoning_effort") if isinstance(metadata.get("reasoning_effort"), str) else None,
                    usage=metadata.get("usage") if isinstance(metadata.get("usage"), dict) else None,
                    estimated_cost_usd=float(metadata.get("estimated_cost_usd")) if isinstance(metadata.get("estimated_cost_usd"), (int, float)) else None,
                    safety_scan_pass=metadata.get("safety_scan_pass") if isinstance(metadata.get("safety_scan_pass"), bool) else None,
                    safety_scan_issues=safety_issues,
                    backend_config=backend_config,
                    notes=candidate_notes,
                )
            )

    existing_dirs = _candidate_dirs(
        config,
        "existing_candidate_dirs",
        [
            "external/KernelBench/solutions",
            "external/KernelBench/candidates",
            "external/KernelBench/generated",
            "external/KernelBench/results/samples",
            "external/KernelBench/results/generations",
        ],
    )
    add_candidate_files(
        directories=existing_dirs,
        candidate_source_type="existing_kernelbench_solution",
        generation_method="manual",
        id_label="existing_solution",
        default_history=("existing_kernelbench_solution",),
        default_notes="Existing KernelBench candidate/solution file discovered under configured external directories.",
    )

    generated_dirs = _candidate_dirs(config, "generated_candidate_dirs", ["generated_candidates"])
    add_candidate_files(
        directories=generated_dirs,
        candidate_source_type="generated_candidate",
        generation_method="llm_plan_impl",
        id_label="generated",
        default_history=("generated_candidate_file",),
        default_notes="Generated candidate file. Must define ModelNew and must not include fallback to the original PyTorch implementation.",
        is_generated_candidate=True,
    )

    manual_dirs = _candidate_dirs(config, "manual_candidate_dirs", ["manual_seed_candidates", "manual_seed", "manual_candidates"])
    add_candidate_files(
        directories=manual_dirs,
        candidate_source_type="manual_seed_candidate",
        generation_method="manual",
        id_label="manual_seed",
        default_history=("manual_seed_candidate",),
        default_notes="Manual seed candidate. Report separately because manual seeds can bias experiments.",
        is_manual_seed=True,
    )

    future_llm_dirs = _candidate_dirs(config, "future_llm_candidate_dirs", ["future_llm_generated_candidates", "llm_generated_candidates"])
    add_candidate_files(
        directories=future_llm_dirs,
        candidate_source_type="future_llm_generated_candidate",
        generation_method="llm_plan_impl",
        id_label="future_llm",
        default_history=("future_llm_generated_candidate",),
        default_notes="Future real LLM-generated candidate. Prompt and response hashes should be recorded when available.",
        is_generated_candidate=True,
    )
    return candidates



def select_candidates(candidates: list[CandidateSpec], max_candidates: int, policy: str = "diversity_first") -> list[CandidateSpec]:
    if policy == "enumeration_order":
        return candidates[:max_candidates]
    if policy != "diversity_first":
        raise ValueError(f"unsupported candidate selection policy: {policy}")
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            CANDIDATE_SELECTION_PRIORITIES.get(item[1].candidate_source_type, 100),
            item[0],
            item[1].candidate_id,
        ),
    )
    return [candidate for _, candidate in ranked[:max_candidates]]


def select_candidates_for_collection(
    *,
    task_id: int,
    candidates: list[CandidateSpec],
    max_candidates: int,
    policy: str,
    existing_unique_keys: set[str] | None = None,
    skip_existing: bool = False,
) -> tuple[list[CandidateSpec], list[CandidateSpec]]:
    """Select up to max_candidates new candidates, backfilling past manifest duplicates."""
    ranked = select_candidates(candidates, len(candidates), policy)
    if not skip_existing:
        return ranked[:max_candidates], []
    existing = existing_unique_keys or set()
    selected: list[CandidateSpec] = []
    skipped_existing: list[CandidateSpec] = []
    for candidate in ranked:
        if _candidate_unique_key(task_id, candidate) in existing:
            skipped_existing.append(candidate)
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected, skipped_existing


def _planned_rows(level: int, tasks: list[int], config: dict[str, Any], max_candidates_per_task: int, run_ncu: bool, selection_policy: str = "diversity_first", existing_unique_keys: set[str] | None = None, skip_existing: bool = False) -> list[dict[str, Any]]:
    rows = []
    existing = existing_unique_keys or set()
    for task_id in tasks:
        candidates = enumerate_candidates(level, task_id, config)
        ranked = select_candidates(candidates, len(candidates), selection_policy)
        planned: list[tuple[CandidateSpec, bool, bool, str | None]] = []
        new_count = 0
        for candidate in ranked:
            unique_key = _candidate_unique_key(task_id, candidate)
            already_existing = unique_key in existing
            if skip_existing and already_existing:
                planned.append((candidate, already_existing, False, "existing_manifest"))
                continue
            planned.append((candidate, already_existing, True, None))
            new_count += 1
            if new_count >= max_candidates_per_task:
                break
            if not skip_existing and len(planned) >= max_candidates_per_task:
                break
        source_counts: dict[str, int] = {}
        config_counts: dict[str, int] = {}
        for candidate, _, _, _ in planned:
            source_counts[candidate.source_hash] = source_counts.get(candidate.source_hash, 0) + 1
            config_counts[candidate.candidate_config_hash] = config_counts.get(candidate.candidate_config_hash, 0) + 1
        for candidate, already_existing, will_collect, skip_reason in planned:
            unique_key = _candidate_unique_key(task_id, candidate)
            row = candidate.dry_run_record(
                task_id=task_id,
                level=level,
                duplicate_source_hash=source_counts[candidate.source_hash] > 1,
                duplicate_candidate_config=config_counts[candidate.candidate_config_hash] > 1,
                run_ncu=run_ncu,
                already_existing=already_existing,
                skip_reason=skip_reason,
            )
            row["unique_key"] = unique_key
            row["will_collect"] = will_collect
            row["oom_timeout_risk"] = _task_risk_note(task_id)
            row["ncu_metrics_config"] = _ncu_metrics_config(config)
            row["ncu_metric_profile_version"] = _ncu_metric_profile_version(config)
            rows.append(row)
    return rows


def _write_dry_run_report(
    *,
    level: int,
    tasks: list[int],
    config: dict[str, Any],
    max_candidates_per_task: int,
    run_ncu: bool,
    out_path: Path,
    selection_policy: str = "diversity_first",
    existing_unique_keys: set[str] | None = None,
    skip_existing: bool = False,
) -> None:
    rows = _planned_rows(level, tasks, config, max_candidates_per_task, run_ncu, selection_policy, existing_unique_keys, skip_existing)
    source_counts: dict[str, int] = {}
    generation_counts: dict[str, int] = {}
    task_counts: dict[int, int] = {}
    duplicate_source_hash_rows = 0
    duplicate_candidate_config_rows = 0
    already_existing_rows = 0
    will_collect_rows = 0
    new_source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["candidate_source_type"]] = source_counts.get(row["candidate_source_type"], 0) + 1
        generation_counts[row["generation_method"]] = generation_counts.get(row["generation_method"], 0) + 1
        task_counts[int(row["task_id"])] = task_counts.get(int(row["task_id"]), 0) + 1
        duplicate_source_hash_rows += int(bool(row["duplicate_source_hash_in_task"]))
        duplicate_candidate_config_rows += int(bool(row["duplicate_candidate_config_hash_in_task"]))
        already_existing_rows += int(bool(row.get("already_existing_candidate")))
        will_collect_rows += int(bool(row.get("will_collect", True)))
        if row.get("will_collect", True):
            new_source_counts[row["candidate_source_type"]] = new_source_counts.get(row["candidate_source_type"], 0) + 1

    available_source_types = sorted(source_counts)
    repeated_baseline_only = bool(available_source_types) and available_source_types == ["pytorch_eager_baseline"]
    baseline_compile_only = bool(available_source_types) and set(available_source_types).issubset(
        {"pytorch_eager_baseline", "torch_compile_baseline"}
    )
    enough_for_baseline_sweep = bool(available_source_types) and len(tasks) >= 30
    enough_for_100_300 = len(rows) >= 100 and not repeated_baseline_only and not baseline_compile_only

    json_path = Path("results/profile_pairs/phase_04_candidate_source_dry_run.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"generated_at": utc_timestamp(), "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase 04 Candidate Source Dry Run",
        "",
        f"- Generated: `{utc_timestamp()}`",
        f"- Level: `{level}`",
        f"- Tasks: `{len(tasks)}`",
        f"- Requested max candidates per task: `{max_candidates_per_task}`",
        f"- Candidate selection policy: `{selection_policy}`",
        f"- NCU metrics config: `{_ncu_metrics_config(config)}`",
        f"- NCU metric profile version: `{_ncu_metric_profile_version(config)}`",
        f"- Planned selected candidates: `{len(rows)}`",
        f"- Candidate source types: `{', '.join(available_source_types) or 'none'}`",
        f"- Generation methods: `{', '.join(sorted(generation_counts)) or 'none'}`",
        f"- Repeats only PyTorch eager baseline: `{repeated_baseline_only}`",
        f"- Includes torch_compile candidate: `{'torch_compile_baseline' in available_source_types}`",
        f"- Duplicate source_hash rows: `{duplicate_source_hash_rows}`",
        f"- Duplicate candidate_config_hash rows: `{duplicate_candidate_config_rows}`",
        f"- Baseline/torch.compile only: `{baseline_compile_only}`",
        f"- Already existing candidates: `{already_existing_rows}`",
        f"- New candidates to collect: `{will_collect_rows}`",
        f"- Skip existing enabled: `{skip_existing}`",
        f"- Candidate diversity sufficient for 30x1 baseline sweep: `{enough_for_baseline_sweep}`",
        f"- Candidate diversity sufficient for 100-300 collection: `{enough_for_100_300}`",
        f"- Structured dry-run JSON: `{json_path}`",
        "",
        "## Source Distribution",
        "",
        "| Source type | Planned count |",
        "|---|---:|",
    ]
    if source_counts:
        for kind, count in sorted(source_counts.items()):
            lines.append(f"| `{kind}` | {count} |")
    else:
        lines.append("| missing | 0 |")
    lines.extend(["", "## New Candidate Source Distribution", "", "| Source type | New planned count |", "|---|---:|"])
    if new_source_counts:
        for kind, count in sorted(new_source_counts.items()):
            lines.append(f"| `{kind}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Generation Method Distribution", "", "| Generation method | Planned count |", "|---|---:|"])
    if generation_counts:
        for kind, count in sorted(generation_counts.items()):
            lines.append(f"| `{kind}` | {count} |")
    else:
        lines.append("| missing | 0 |")
    lines.extend(
        [
            "",
            "## Per-Candidate Plan",
            "",
            "| Task | Count | Source type | Generation | Candidate ID | Existing | Will collect | Skip reason | Source hash dup | Config hash dup | OOM/timeout risk | Source path | Source hash | Backend config hash | Candidate config hash | Correctness required | NCU if correct |",
            "|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {task_id} | {count} | `{candidate_source_type}` | `{generation_method}` | `{candidate_id}` | {already_existing_candidate} | {will_collect} | `{skip_reason}` | {dup_src} | {dup_cfg} | `{oom_timeout_risk}` | `{candidate_source_path}` | `{source_hash}` | `{backend_config_hash}` | `{candidate_config_hash}` | {correctness_required} | {ncu_profile_required_if_correct} |".format(
                count=task_counts[int(row["task_id"])],
                dup_src=row["duplicate_source_hash_in_task"],
                dup_cfg=row["duplicate_candidate_config_hash_in_task"],
                **row,
            )
        )
    lines.extend(["", "## Interpretation", ""])
    if repeated_baseline_only:
        lines.append("- Current collection would repeat only the PyTorch eager baseline. Do not run 100-300 pairs until candidate sources are expanded.")
    elif baseline_compile_only:
        lines.append("- Current sources are PyTorch eager and torch.compile only. This is enough for a small smoke test, but not enough for 100-300 research pairs.")
    elif len(available_source_types) < 2:
        lines.append("- Candidate source diversity is limited. Add generated or manual candidate sources before scale-up.")
    else:
        lines.append("- Candidate sources include at least two source types. Run a bounded smoke test before any larger collection.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    legacy_path = Path("reports/candidate_source_audit.md")
    if legacy_path != out_path:
        legacy_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")


def _candidate_record_common(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_key": candidate.resume_key,
        "parent_candidate_id": candidate.parent_candidate_id,
        "candidate_source_path": str(candidate.source_path),
        "source_hash": candidate.source_hash,
        "backend_config_hash": candidate.backend_config_hash,
        "candidate_config_hash": candidate.candidate_config_hash,
        "candidate_source_type": candidate.candidate_source_type,
        "backend": candidate.backend,
        "generation_method": candidate.generation_method,
        "transformation_history": list(candidate.transformation_history),
        "is_baseline_candidate": candidate.is_baseline_candidate,
        "is_generated_candidate": candidate.is_generated_candidate,
        "is_manual_seed": candidate.is_manual_seed,
        "notes": candidate.notes,
        "prompt_hash": candidate.prompt_hash,
        "response_hash": candidate.response_hash,
        "model_name": candidate.model_name,
        "provider": candidate.provider,
        "temperature": candidate.temperature,
        "reasoning_effort": candidate.reasoning_effort,
        "usage": candidate.usage,
        "estimated_cost_usd": candidate.estimated_cost_usd,
        "safety_scan_pass": candidate.safety_scan_pass,
        "safety_scan_issues": list(candidate.safety_scan_issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect cheap/high-fidelity profile-pair records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-candidates-per-task", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", help="Output JSONL path.")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List planned candidate sources without executing timing or NCU.")
    parser.add_argument("--candidate-selection", choices=["diversity_first", "enumeration_order"], default="diversity_first", help="Candidate selection policy for max-candidates-per-task.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip candidates already present in an existing unique manifest.")
    parser.add_argument("--existing-manifest", help="Unique manifest JSONL used by --skip-existing.")
    parser.add_argument("--ncu-metrics", default=None, help="Metric config YAML for selected Nsight Compute profiling.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.ncu_metrics:
        config["ncu_metrics_config"] = args.ncu_metrics
        config.pop("ncu_metric_profile_version", None)
    level = int(config.get("level", 1))
    tasks = [int(task) for task in config.get("tasks", [1])]
    num_warmup = int(config.get("num_warmup", 10))
    num_repeats = int(config.get("num_repeats", 50))
    run_ncu = bool(config.get("run_ncu", True))
    prefix = str(config.get("experiment_prefix", "phase04_collect"))
    ncu_metrics_config = _ncu_metrics_config(config)
    ncu_metric_profile_version = _ncu_metric_profile_version(config)
    stamp = utc_timestamp().replace(":", "").replace("+", "Z")
    out_path = Path(args.output) if args.output else ensure_dir("results/profile_pairs") / f"{prefix}_profile_pairs_l{level}_{stamp}.jsonl"
    existing_unique_keys = _load_existing_unique_keys(args.existing_manifest)

    if args.dry_run:
        report_path = Path("reports/phase_04_candidate_source_dry_run.md")
        _write_dry_run_report(
            level=level,
            tasks=tasks,
            config=config,
            max_candidates_per_task=args.max_candidates_per_task,
            run_ncu=run_ncu,
            out_path=report_path,
            selection_policy=args.candidate_selection,
            existing_unique_keys=existing_unique_keys,
            skip_existing=args.skip_existing,
        )
        print(report_path)
        return 0

    seen = set() if args.no_resume else _existing_resume_keys(out_path)
    report_lines = ["# Profile Pair Collection Smoke", ""]
    exit_code = 0

    for task_id in tasks:
        candidates = enumerate_candidates(level, task_id, config)
        selected_candidates, skipped_existing_candidates = select_candidates_for_collection(
            task_id=task_id,
            candidates=candidates,
            max_candidates=args.max_candidates_per_task,
            policy=args.candidate_selection,
            existing_unique_keys=existing_unique_keys,
            skip_existing=args.skip_existing,
        )
        if args.max_candidates_per_task > len(candidates):
            report_lines.append(f"- Level {level} task {task_id}: only {len(candidates)} safe candidate source(s) available for this smoke run.")
        for skipped in skipped_existing_candidates:
            report_lines.append(f"- Skipped task {task_id}: source={skipped.candidate_source_type} candidate={skipped.candidate_id} unique_key already present in existing manifest.")
        for cand_idx, candidate in enumerate(selected_candidates):
            if candidate.resume_key in seen:
                report_lines.append(f"- Skipped task {task_id} candidate {cand_idx}: candidate_config_hash already present.")
                continue
            start = time.perf_counter()
            experiment_id = f"{prefix}_pair_l{level}_t{task_id}_{candidate.candidate_source_type}_{int(time.time())}"
            baseline_out = ensure_dir("results/baselines") / f"collect_target_{experiment_id}.jsonl"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.device)
            baseline_cmd = [
                sys.executable,
                "scripts/run_kernelbench_baseline.py",
                "--level",
                str(level),
                "--task",
                str(task_id),
                "--backend",
                candidate.backend,
                "--num-warmup",
                str(num_warmup),
                "--num-repeats",
                str(num_repeats),
                "--device",
                "0",
                "--output",
                str(baseline_out),
            ]
            if candidate.backend == "candidate":
                baseline_cmd.extend(["--candidate-source", str(candidate.source_path)])
            baseline_run = run_command(baseline_cmd, cwd=str(REPO_ROOT), env=env, timeout=args.timeout)
            baseline_records = read_jsonl(baseline_out) if baseline_out.exists() else []
            baseline = baseline_records[-1] if baseline_records else {
                "correctness_pass": False,
                "correctness_error": baseline_run.stderr or baseline_run.error or "baseline produced no JSONL record",
                "timing_time_sec": baseline_run.duration_sec,
            }
            cheap = extract_static_features(candidate.source_path)
            profile = {
                "profiling_status": "not_run",
                "profiling_time_sec": 0.0,
                "parsed_metrics": {},
                "metrics_unavailable": [],
                "failure_reason": None,
            }
            if run_ncu and baseline.get("correctness_pass"):
                profile = run_ncu_profile(
                    baseline_cmd,
                    requested_metrics=load_requested_metrics(ncu_metrics_config),
                    output_dir="results/profiles",
                    experiment_id=experiment_id,
                    timeout=args.timeout,
                    env=env,
                )
            elif run_ncu:
                profile["profiling_status"] = "skipped"
                profile["failure_reason"] = "correctness failed or baseline unavailable"

            failure_reason = baseline.get("correctness_error") or profile.get("failure_reason")
            record = {
                "experiment_id": experiment_id,
                "task_id": task_id,
                "level": level,
                **_candidate_record_common(candidate),
                "failure_reason": failure_reason,
                "oom_timeout_risk": _task_risk_note(task_id),
                "optimization_plan_text": None,
                "correctness_pass": bool(baseline.get("correctness_pass")),
                "correctness_error": baseline.get("correctness_error"),
                "latency_stats": {
                    "mean_ms": baseline.get("latency_mean_ms"),
                    "median_ms": baseline.get("latency_median_ms"),
                    "std_ms": baseline.get("latency_std_ms"),
                    "p05_ms": baseline.get("latency_p05_ms"),
                    "p95_ms": baseline.get("latency_p95_ms"),
                    "num_warmup": num_warmup,
                    "num_repeats": num_repeats,
                    "raw_latency_file": baseline.get("raw_latency_file"),
                },
                "cheap_features": cheap,
                "ncu_metric_profile_version": ncu_metric_profile_version,
                "ncu_metrics_config": ncu_metrics_config,
                "high_fidelity_profile": {
                    "ncu_metric_profile_version": ncu_metric_profile_version,
                    "ncu_metrics_config": ncu_metrics_config,
                    "metrics_requested": profile.get("metrics_requested", []),
                    "metrics_available": profile.get("metrics_available", []),
                    "selected_ncu_metrics": _mean_metrics(profile.get("parsed_metrics", {})),
                    "unavailable_metrics": profile.get("metrics_unavailable", []),
                    "profiling_time_sec": profile.get("profiling_time_sec"),
                    "profiling_status": profile.get("profiling_status"),
                    "failure_reason": profile.get("failure_reason"),
                    "raw_output_path": profile.get("raw_output_path"),
                },
                "predictor": {
                    "predicted_metrics": {},
                    "uncertainty": None,
                    "used_full_profile": bool(run_ncu and baseline.get("correctness_pass")),
                },
                "cost_accounting": {
                    "llm_time_sec": 0.0,
                    "compile_time_sec": 0.0,
                    "timing_time_sec": baseline.get("timing_time_sec", baseline_run.duration_sec),
                    "static_feature_time_sec": cheap.get("static_feature_time_sec"),
                    "ncu_profile_time_sec": profile.get("profiling_time_sec", 0.0),
                    "predictor_inference_time_sec": 0.0,
                    "predictor_update_time_sec": 0.0,
                    "total_wall_clock_sec": time.perf_counter() - start,
                },
                "environment": collect_environment_metadata(args.device, REPO_ROOT),
            }
            append_jsonl(out_path, record)
            seen.add(candidate.resume_key)
            if not record["correctness_pass"] or profile.get("profiling_status") not in SUCCESS_STATUSES | {"not_run"}:
                exit_code = 1
            report_lines.append(
                f"- Task {task_id} candidate {cand_idx}: source={candidate.candidate_source_type} correctness={record['correctness_pass']} "
                f"profile={profile.get('profiling_status')} output=`{out_path}`"
            )

    report_path = Path("reports/profile_pair_collection_smoke.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(out_path)
    print(report_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
