from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass
class KernelBenchTask:
    level: int
    task_id: int
    path: Path
    module: ModuleType

    @property
    def name(self) -> str:
        return self.path.stem

    def make_model(self) -> Any:
        init_inputs = self.get_init_inputs()
        return self.module.Model(*init_inputs)

    def get_inputs(self) -> list[Any]:
        return list(self.module.get_inputs())

    def get_init_inputs(self) -> list[Any]:
        if hasattr(self.module, "get_init_inputs"):
            return list(self.module.get_init_inputs())
        return []


def repo_root_from(start: str | Path | None = None) -> Path:
    start_path = Path(start or Path.cwd()).resolve()
    for path in [start_path, *start_path.parents]:
        if (path / "external" / "KernelBench").exists():
            return path
    return start_path


def kernelbench_root(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else repo_root_from()
    return root / "external" / "KernelBench"


def level_dir(level: int, repo_root: str | Path | None = None) -> Path:
    return kernelbench_root(repo_root) / "KernelBench" / f"level{level}"


def list_tasks(level: int, repo_root: str | Path | None = None) -> list[Path]:
    directory = level_dir(level, repo_root)
    if not directory.exists():
        raise FileNotFoundError(f"KernelBench level directory not found: {directory}")
    tasks = [path for path in directory.glob("*.py") if re.match(r"^\d+_", path.name)]
    return sorted(tasks, key=lambda path: int(path.name.split("_", 1)[0]))


def find_task_path(level: int, task_id: int, repo_root: str | Path | None = None) -> Path:
    for path in list_tasks(level, repo_root):
        if int(path.name.split("_", 1)[0]) == int(task_id):
            return path
    raise FileNotFoundError(f"KernelBench level {level} task {task_id} not found")


def import_module_from_path(path: str | Path, module_name: str | None = None) -> ModuleType:
    src_path = Path(path).resolve()
    name = module_name or f"profbridge_kernelbench_{src_path.stem}_{abs(hash(src_path))}"
    spec = importlib.util.spec_from_file_location(name, src_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import module from {src_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_task(level: int, task_id: int, repo_root: str | Path | None = None) -> KernelBenchTask:
    path = find_task_path(level, task_id, repo_root)
    module = import_module_from_path(path)
    for attr in ["Model", "get_inputs"]:
        if not hasattr(module, attr):
            raise AttributeError(f"{path} missing required KernelBench attribute {attr}")
    return KernelBenchTask(level=int(level), task_id=int(task_id), path=path, module=module)


def load_candidate_model(path: str | Path, class_name: str = "ModelNew") -> Any:
    module = import_module_from_path(path)
    if not hasattr(module, class_name):
        raise AttributeError(f"{path} missing {class_name}")
    init_inputs = list(module.get_init_inputs()) if hasattr(module, "get_init_inputs") else []
    return getattr(module, class_name)(*init_inputs)
