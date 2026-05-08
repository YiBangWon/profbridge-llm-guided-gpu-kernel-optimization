from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class CommandResult:
    command: list[str] | str
    returncode: int | None
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None


def run_command(
    command: Sequence[str] | str,
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
) -> CommandResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            shell=shell,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return CommandResult(
            command=list(command) if not isinstance(command, str) else command,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_sec=time.perf_counter() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=list(command) if not isinstance(command, str) else command,
            returncode=None,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            duration_sec=time.perf_counter() - start,
            timed_out=True,
            error=f"timeout after {timeout} seconds",
        )
    except Exception as exc:  # pragma: no cover - defensive for hostile envs
        return CommandResult(
            command=list(command) if not isinstance(command, str) else command,
            returncode=None,
            stdout="",
            stderr="",
            duration_sec=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
