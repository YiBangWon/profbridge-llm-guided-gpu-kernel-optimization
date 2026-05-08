from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PtxasFeatures:
    registers_per_thread: int | None = None
    shared_memory_bytes: int | None = None
    constant_memory_bytes: int | None = None
    local_memory_bytes: int | None = None
    spill_stores_bytes: int | None = None
    spill_loads_bytes: int | None = None
    stack_frame_bytes: int | None = None
    entry_functions: list[str] | None = None
    parse_errors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ptxas_registers_per_thread": self.registers_per_thread,
            "ptxas_shared_memory_bytes": self.shared_memory_bytes,
            "ptxas_constant_memory_bytes": self.constant_memory_bytes,
            "ptxas_local_memory_bytes": self.local_memory_bytes,
            "ptxas_spill_stores_bytes": self.spill_stores_bytes,
            "ptxas_spill_loads_bytes": self.spill_loads_bytes,
            "ptxas_stack_frame_bytes": self.stack_frame_bytes,
            "ptxas_entry_functions": self.entry_functions or [],
            "ptxas_parse_errors": self.parse_errors or [],
        }


def _max_seen(current: int | None, value: int | None) -> int | None:
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)


def parse_ptxas_output(text: str) -> dict[str, Any]:
    features = PtxasFeatures(entry_functions=[], parse_errors=[])
    for line in text.splitlines():
        entry = re.search(r"Compiling entry function '([^']+)'", line)
        if entry:
            features.entry_functions.append(entry.group(1))
        reg = re.search(r"Used\s+(\d+)\s+registers", line)
        if reg:
            features.registers_per_thread = _max_seen(features.registers_per_thread, int(reg.group(1)))
        smem = re.search(r"(\d+)\s+bytes\s+smem", line)
        if smem:
            features.shared_memory_bytes = _max_seen(features.shared_memory_bytes, int(smem.group(1)))
        lmem = re.search(r"(\d+)\s+bytes\s+lmem", line)
        if lmem:
            features.local_memory_bytes = _max_seen(features.local_memory_bytes, int(lmem.group(1)))
        cmems = [int(match.group(1)) for match in re.finditer(r"(\d+)\s+bytes\s+cmem\[\d+\]", line)]
        if cmems:
            features.constant_memory_bytes = _max_seen(features.constant_memory_bytes, sum(cmems))
        stack = re.search(r"(\d+)\s+bytes\s+stack\s+frame", line)
        if stack:
            features.stack_frame_bytes = _max_seen(features.stack_frame_bytes, int(stack.group(1)))
        stores = re.search(r"(\d+)\s+bytes\s+spill\s+stores", line)
        if stores:
            features.spill_stores_bytes = _max_seen(features.spill_stores_bytes, int(stores.group(1)))
        loads = re.search(r"(\d+)\s+bytes\s+spill\s+loads", line)
        if loads:
            features.spill_loads_bytes = _max_seen(features.spill_loads_bytes, int(loads.group(1)))
    return features.to_dict()


def synthetic_parser_test() -> bool:
    sample = """
ptxas info    : Compiling entry function '_Z6kernelPf' for 'sm_80'
ptxas info    : Function properties for _Z6kernelPf
    16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 64 registers, 1024 bytes smem, 384 bytes cmem[0], 8 bytes cmem[2]
"""
    parsed = parse_ptxas_output(sample)
    return (
        parsed["ptxas_registers_per_thread"] == 64
        and parsed["ptxas_shared_memory_bytes"] == 1024
        and parsed["ptxas_spill_stores_bytes"] == 8
        and parsed["ptxas_spill_loads_bytes"] == 4
    )
