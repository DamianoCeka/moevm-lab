"""Read-only machine and configuration diagnostics for MoEVM.

This module deliberately has no accelerator runtime dependency.  It inspects a
validated experiment configuration, asks the operating system for host memory
and disk capacity, and can optionally call ``nvidia-smi``.  It never opens a
CUDA context, writes a file, downloads a model, or imports a benchmark script.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import ExperimentConfig

_MIB = 1024 * 1024
ProbeStatus = Literal["available", "unavailable", "error", "timeout"]


@dataclass(frozen=True, slots=True)
class MemoryLedger:
    """Logical expert-weight capacity derived from an experiment configuration.

    The byte values describe the simulator's configured expert payloads.  They
    are not measurements of a checkpoint, live process, or physical transfer.
    In particular, the routed payload is the total logical expert payload for
    one token across all MoE layers before cache hits are considered.
    """

    expert_bytes: int
    layers: int
    experts_per_layer: int
    top_k: int
    total_experts: int
    vram_cache_bytes: int
    ram_cache_bytes: int
    vram_cache_slots: int
    ram_cache_slots: int
    vram_cache_remainder_bytes: int
    ram_cache_remainder_bytes: int
    all_expert_logical_bytes: int
    routed_expert_accesses_per_token: int
    per_layer_logical_routed_expert_bytes: int
    per_token_logical_routed_expert_bytes: int


@dataclass(frozen=True, slots=True)
class SystemMemory:
    """Host physical-memory capacity observed without loading an ML runtime."""

    status: ProbeStatus
    source: str
    total_bytes: int | None
    available_bytes: int | None
    detail: str | None = None

    @property
    def used_bytes(self) -> int | None:
        """Best-effort used physical memory, when total and available are known."""
        if self.total_bytes is None or self.available_bytes is None:
            return None
        return max(0, self.total_bytes - self.available_bytes)


@dataclass(frozen=True, slots=True)
class DiskSpace:
    """Free space for the volume that contains a caller-selected path."""

    status: ProbeStatus
    requested_path: str
    inspected_path: str | None
    total_bytes: int | None
    used_bytes: int | None
    free_bytes: int | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """One GPU row reported by ``nvidia-smi`` without creating a CUDA context."""

    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    driver_version: str


@dataclass(frozen=True, slots=True)
class GpuProbe:
    """Result of the optional ``nvidia-smi`` probe."""

    status: ProbeStatus
    gpus: tuple[GpuInfo, ...]
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class MachineReport:
    """A CLI-ready, read-only snapshot of config and host capacity."""

    ledger: MemoryLedger
    system_memory: SystemMemory
    disk: DiskSpace
    gpu: GpuProbe | None


def build_memory_ledger(config: ExperimentConfig) -> MemoryLedger:
    """Return logical expert-weight accounting for a validated configuration."""
    config.validate()
    expert_bytes = config.model.expert_size_bytes
    vram_cache_bytes = config.hardware.vram_cache_bytes
    ram_cache_bytes = config.hardware.ram_cache_bytes
    vram_cache_slots, vram_remainder = divmod(vram_cache_bytes, expert_bytes)
    ram_cache_slots, ram_remainder = divmod(ram_cache_bytes, expert_bytes)
    total_experts = config.model.layers * config.model.experts_per_layer
    routed_accesses = config.model.layers * config.model.top_k

    return MemoryLedger(
        expert_bytes=expert_bytes,
        layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        top_k=config.model.top_k,
        total_experts=total_experts,
        vram_cache_bytes=vram_cache_bytes,
        ram_cache_bytes=ram_cache_bytes,
        vram_cache_slots=vram_cache_slots,
        ram_cache_slots=ram_cache_slots,
        vram_cache_remainder_bytes=vram_remainder,
        ram_cache_remainder_bytes=ram_remainder,
        all_expert_logical_bytes=total_experts * expert_bytes,
        routed_expert_accesses_per_token=routed_accesses,
        per_layer_logical_routed_expert_bytes=config.model.top_k * expert_bytes,
        per_token_logical_routed_expert_bytes=routed_accesses * expert_bytes,
    )


def _parse_proc_meminfo(contents: str) -> tuple[int, int]:
    """Extract total and available bytes from Linux ``/proc/meminfo`` text."""
    values_kib: dict[str, int] = {}
    for line in contents.splitlines():
        name, separator, rest = line.partition(":")
        if not separator:
            continue
        fields = rest.split()
        if not fields:
            continue
        try:
            values_kib[name] = int(fields[0])
        except ValueError:
            continue

    total_kib = values_kib.get("MemTotal")
    available_kib = values_kib.get("MemAvailable")
    if total_kib is None or available_kib is None:
        raise ValueError("/proc/meminfo did not contain MemTotal and MemAvailable")
    if total_kib <= 0 or available_kib < 0:
        raise ValueError("/proc/meminfo contained invalid memory values")
    return total_kib * 1024, available_kib * 1024


def _probe_windows_memory() -> tuple[int, int]:
    """Read physical-memory totals through ``GlobalMemoryStatusEx`` on Windows."""

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    kernel32 = ctypes.windll.kernel32
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _probe_sysconf_memory() -> tuple[int, int]:
    """Read POSIX physical and available pages when the platform exposes them."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = os.sysconf("SC_PHYS_PAGES")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    if page_size <= 0 or total_pages <= 0 or available_pages < 0:
        raise ValueError("sysconf returned invalid physical-memory values")
    return int(page_size * total_pages), int(page_size * available_pages)


def probe_system_memory() -> SystemMemory:
    """Return best-effort host RAM totals without requiring third-party packages."""
    if sys.platform == "win32":
        try:
            total_bytes, available_bytes = _probe_windows_memory()
        except (AttributeError, OSError) as exc:
            return SystemMemory(
                status="error",
                source="GlobalMemoryStatusEx",
                total_bytes=None,
                available_bytes=None,
                detail=str(exc),
            )
        return SystemMemory(
            status="available",
            source="GlobalMemoryStatusEx",
            total_bytes=total_bytes,
            available_bytes=available_bytes,
        )

    proc_meminfo = Path("/proc/meminfo")
    try:
        if proc_meminfo.is_file():
            total_bytes, available_bytes = _parse_proc_meminfo(
                proc_meminfo.read_text(encoding="utf-8")
            )
            return SystemMemory(
                status="available",
                source="/proc/meminfo",
                total_bytes=total_bytes,
                available_bytes=available_bytes,
            )
    except (OSError, ValueError) as exc:
        proc_error = str(exc)
    else:
        proc_error = None

    try:
        total_bytes, available_bytes = _probe_sysconf_memory()
    except (AttributeError, OSError, ValueError) as exc:
        detail = str(exc)
        if proc_error:
            detail = f"/proc/meminfo: {proc_error}; sysconf: {detail}"
        return SystemMemory(
            status="unavailable",
            source="sysconf",
            total_bytes=None,
            available_bytes=None,
            detail=detail,
        )
    return SystemMemory(
        status="available",
        source="sysconf",
        total_bytes=total_bytes,
        available_bytes=available_bytes,
    )


def _nearest_existing_path(path: Path) -> Path | None:
    """Find the mounted ancestor of a selected path without creating anything."""
    candidate = path
    while True:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            return None
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def probe_disk(path: str | Path) -> DiskSpace:
    """Return free capacity for ``path`` or its nearest existing ancestor.

    A model directory may not exist yet.  In that case the report still exposes
    the free space on the volume where it would be created and records the
    inspected ancestor separately from the requested path.
    """
    requested = Path(path).expanduser()
    inspected = _nearest_existing_path(requested)
    if inspected is None:
        return DiskSpace(
            status="unavailable",
            requested_path=str(requested),
            inspected_path=None,
            total_bytes=None,
            used_bytes=None,
            free_bytes=None,
            detail="no existing ancestor was available for disk inspection",
        )
    try:
        usage = shutil.disk_usage(inspected)
    except OSError as exc:
        return DiskSpace(
            status="error",
            requested_path=str(requested),
            inspected_path=str(inspected),
            total_bytes=None,
            used_bytes=None,
            free_bytes=None,
            detail=str(exc),
        )
    return DiskSpace(
        status="available",
        requested_path=str(requested),
        inspected_path=str(inspected),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
    )


def _parse_nvidia_smi_rows(output: str) -> tuple[GpuInfo, ...]:
    """Parse the stable CSV layout requested by :func:`probe_nvidia_smi`."""
    gpus: list[GpuInfo] = []
    for line in (item.strip() for item in output.splitlines()):
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(
                "nvidia-smi returned an unexpected CSV row "
                f"with {len(fields)} fields: {line!r}"
            )
        name, total_mib, free_mib, driver_version = fields
        try:
            total_vram_bytes = int(total_mib) * _MIB
            free_vram_bytes = int(free_mib) * _MIB
        except ValueError as exc:
            raise ValueError(
                f"nvidia-smi returned non-integer memory values: {line!r}"
            ) from exc
        if (
            not name
            or not driver_version
            or total_vram_bytes < 0
            or free_vram_bytes < 0
            or free_vram_bytes > total_vram_bytes
        ):
            raise ValueError(f"nvidia-smi returned invalid GPU values: {line!r}")
        gpus.append(
            GpuInfo(
                index=len(gpus),
                name=name,
                total_vram_bytes=total_vram_bytes,
                free_vram_bytes=free_vram_bytes,
                driver_version=driver_version,
            )
        )
    if not gpus:
        raise ValueError("nvidia-smi returned no GPU rows")
    return tuple(gpus)


def probe_nvidia_smi(*, timeout_seconds: float = 2.0) -> GpuProbe:
    """Optionally inspect NVIDIA GPU capacity through a bounded shell-free call."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return GpuProbe(
            status="unavailable",
            gpus=(),
            detail="nvidia-smi was not found on PATH",
        )
    except PermissionError as exc:
        return GpuProbe(status="error", gpus=(), detail=str(exc))
    except subprocess.TimeoutExpired:
        return GpuProbe(
            status="timeout",
            gpus=(),
            detail=f"nvidia-smi did not finish within {timeout_seconds:g} seconds",
        )
    except OSError as exc:
        return GpuProbe(status="error", gpus=(), detail=str(exc))

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        message = f"nvidia-smi exited with status {completed.returncode}"
        if detail:
            message = f"{message}: {detail}"
        return GpuProbe(status="error", gpus=(), detail=message)
    try:
        gpus = _parse_nvidia_smi_rows(completed.stdout)
    except ValueError as exc:
        return GpuProbe(status="error", gpus=(), detail=str(exc))
    return GpuProbe(status="available", gpus=gpus)


def collect_machine_report(
    config: ExperimentConfig,
    *,
    disk_path: str | Path,
    probe_gpu: bool = True,
    gpu_timeout_seconds: float = 2.0,
) -> MachineReport:
    """Collect config accounting plus read-only host probes for a CLI report."""
    return MachineReport(
        ledger=build_memory_ledger(config),
        system_memory=probe_system_memory(),
        disk=probe_disk(disk_path),
        gpu=(
            probe_nvidia_smi(timeout_seconds=gpu_timeout_seconds) if probe_gpu else None
        ),
    )
