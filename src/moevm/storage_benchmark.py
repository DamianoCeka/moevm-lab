from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import random
import socket
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

MIB = 1024 * 1024
DEFAULT_CHUNK_BYTES = 12 * MIB
DEFAULT_ALIGNMENT_BYTES = 4096


def parse_size(value: str) -> int:
    """Parse an integer byte count or a binary/decimal size suffix."""
    text = value.strip().lower()
    suffixes = {
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "b": 1,
    }
    multiplier = 1
    number = text
    for suffix in sorted(suffixes, key=len, reverse=True):
        if text.endswith(suffix):
            multiplier = suffixes[suffix]
            number = text[: -len(suffix)].strip()
            break
    try:
        parsed = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}") from exc
    result = parsed * multiplier
    if not math.isfinite(result) or result <= 0 or not result.is_integer():
        raise argparse.ArgumentTypeError(
            f"size must resolve to a positive whole number of bytes: {value!r}"
        )
    return int(result)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no observations")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _random_aligned_offset(
    rng: random.Random,
    *,
    file_size: int,
    chunk_bytes: int,
    alignment_bytes: int,
) -> int:
    if file_size < chunk_bytes:
        raise ValueError(
            f"target is smaller than one chunk: {file_size} < {chunk_bytes} bytes"
        )
    slots = (file_size - chunk_bytes) // alignment_bytes + 1
    return rng.randrange(slots) * alignment_bytes


def _latency_summary(latencies_ns: list[int]) -> dict[str, float]:
    milliseconds = [value / 1_000_000 for value in latencies_ns]
    return {
        "min": min(milliseconds),
        "mean": sum(milliseconds) / len(milliseconds),
        "p50": _percentile(milliseconds, 0.50),
        "p95": _percentile(milliseconds, 0.95),
        "p99": _percentile(milliseconds, 0.99),
        "max": max(milliseconds),
    }


@dataclass(frozen=True)
class _TargetSnapshot:
    path: Path
    size_bytes: int
    modified_time_ns: int

    @classmethod
    def capture(cls, path: str | os.PathLike[str]) -> _TargetSnapshot:
        resolved = Path(path).expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"benchmark target is not a regular file: {resolved}")
        stat = resolved.stat()
        return cls(
            path=resolved,
            size_bytes=stat.st_size,
            modified_time_ns=stat.st_mtime_ns,
        )

    def assert_unchanged(self) -> None:
        stat = self.path.stat()
        if stat.st_size != self.size_bytes or stat.st_mtime_ns != self.modified_time_ns:
            raise RuntimeError(
                f"target changed while it was being benchmarked: {self.path}"
            )


class _RandomReadHandle(Protocol):
    alignment_bytes: int

    def read_at(self, offset: int) -> int: ...

    def close(self) -> None: ...


class _BufferedReadHandle:
    """Normal read-only file I/O; the operating-system page cache remains active."""

    def __init__(self, path: Path, chunk_bytes: int, alignment_bytes: int) -> None:
        self.alignment_bytes = alignment_bytes
        self._chunk_bytes = chunk_bytes
        self._buffer = bytearray(chunk_bytes)
        # buffering=0 avoids an extra Python buffer, but does not bypass the OS cache.
        self._file = path.open("rb", buffering=0)

    def read_at(self, offset: int) -> int:
        self._file.seek(offset)
        view = memoryview(self._buffer)
        total = 0
        while total < self._chunk_bytes:
            count = self._file.readinto(view[total:])
            if not count:
                raise OSError(
                    f"short read at offset {offset}: {total}/{self._chunk_bytes} bytes"
                )
            total += count
        return self._buffer[0] ^ self._buffer[-1]

    def close(self) -> None:
        self._file.close()


class _WindowsUnbufferedReadHandle:
    """Aligned, read-only Win32 I/O using FILE_FLAG_NO_BUFFERING."""

    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _FILE_FLAG_RANDOM_ACCESS = 0x10000000
    _FILE_STORAGE_INFO_CLASS = 16
    _FILE_BEGIN = 0
    _MEM_COMMIT = 0x1000
    _MEM_RESERVE = 0x2000
    _MEM_RELEASE = 0x8000
    _PAGE_READWRITE = 0x04

    class _FileStorageInfo(ctypes.Structure):
        _fields_ = [
            ("logical_bytes_per_sector", ctypes.c_uint32),
            ("physical_bytes_per_sector_for_atomicity", ctypes.c_uint32),
            ("physical_bytes_per_sector_for_performance", ctypes.c_uint32),
            (
                "filesystem_effective_physical_bytes_per_sector_for_atomicity",
                ctypes.c_uint32,
            ),
            ("flags", ctypes.c_uint32),
            ("byte_offset_for_sector_alignment", ctypes.c_uint32),
            ("byte_offset_for_partition_alignment", ctypes.c_uint32),
        ]

    def __init__(
        self, path: Path, chunk_bytes: int, requested_alignment_bytes: int
    ) -> None:
        if sys.platform != "win32":
            raise ValueError(
                "windows-unbuffered I/O is available only on Windows; "
                "use --io-mode buffered on this platform"
            )
        if chunk_bytes > 0xFFFFFFFF:
            raise ValueError("windows-unbuffered chunks cannot exceed 4 GiB")

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api()
        self._handle: int | None = None
        self._allocation: int | None = None
        self._buffer_address: int | None = None
        self._chunk_bytes = chunk_bytes

        handle = self._kernel32.CreateFileW(
            str(path),
            self._GENERIC_READ,
            self._FILE_SHARE_READ,
            None,
            self._OPEN_EXISTING,
            self._FILE_FLAG_NO_BUFFERING | self._FILE_FLAG_RANDOM_ACCESS,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            self._raise_last_error(f"cannot open {path} with FILE_FLAG_NO_BUFFERING")
        self._handle = handle

        try:
            storage_info = self._FileStorageInfo()
            ok = self._kernel32.GetFileInformationByHandleEx(
                self._handle,
                self._FILE_STORAGE_INFO_CLASS,
                ctypes.byref(storage_info),
                ctypes.sizeof(storage_info),
            )
            if not ok:
                self._raise_last_error(
                    "cannot determine the storage alignment required for safe "
                    "unbuffered I/O"
                )
            sector_sizes = (
                storage_info.logical_bytes_per_sector,
                storage_info.physical_bytes_per_sector_for_atomicity,
                storage_info.physical_bytes_per_sector_for_performance,
                storage_info.filesystem_effective_physical_bytes_per_sector_for_atomicity,
            )
            valid_sector_sizes = [size for size in sector_sizes if size > 0]
            if not valid_sector_sizes:
                raise OSError(
                    "Windows returned no usable sector size; refusing unsafe "
                    "unbuffered I/O"
                )
            required_alignment = max(valid_sector_sizes)
            self.alignment_bytes = math.lcm(
                requested_alignment_bytes, required_alignment
            )
            if chunk_bytes % self.alignment_bytes:
                raise ValueError(
                    "chunk size must be a multiple of the effective Windows "
                    f"unbuffered alignment ({self.alignment_bytes} bytes)"
                )

            allocation_size = chunk_bytes + self.alignment_bytes
            allocation = self._kernel32.VirtualAlloc(
                None,
                allocation_size,
                self._MEM_COMMIT | self._MEM_RESERVE,
                self._PAGE_READWRITE,
            )
            if not allocation:
                self._raise_last_error(
                    f"cannot allocate {allocation_size} bytes for the aligned buffer"
                )
            self._allocation = allocation
            self._buffer_address = (
                (allocation + self.alignment_bytes - 1) // self.alignment_bytes
            ) * self.alignment_bytes
        except Exception:
            self.close()
            raise

    def _configure_api(self) -> None:
        kernel32 = self._kernel32
        kernel32.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.GetFileInformationByHandleEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        kernel32.GetFileInformationByHandleEx.restype = ctypes.c_int
        kernel32.SetFilePointerEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint32,
        ]
        kernel32.SetFilePointerEx.restype = ctypes.c_int
        kernel32.ReadFile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        kernel32.ReadFile.restype = ctypes.c_int
        kernel32.VirtualAlloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.VirtualFree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
        ]
        kernel32.VirtualFree.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

    @staticmethod
    def _raise_last_error(message: str) -> None:
        error = ctypes.get_last_error()
        raise OSError(error, f"{message}: {ctypes.FormatError(error)}")

    def read_at(self, offset: int) -> int:
        if self._handle is None or self._buffer_address is None:
            raise RuntimeError("unbuffered read handle is closed")
        if offset % self.alignment_bytes:
            raise ValueError(
                f"offset {offset} is not aligned to {self.alignment_bytes} bytes"
            )
        new_position = ctypes.c_int64()
        ok = self._kernel32.SetFilePointerEx(
            self._handle,
            offset,
            ctypes.byref(new_position),
            self._FILE_BEGIN,
        )
        if not ok or new_position.value != offset:
            self._raise_last_error(f"cannot seek to offset {offset}")
        bytes_read = ctypes.c_uint32()
        ok = self._kernel32.ReadFile(
            self._handle,
            self._buffer_address,
            self._chunk_bytes,
            ctypes.byref(bytes_read),
            None,
        )
        if not ok:
            self._raise_last_error(f"unbuffered read failed at offset {offset}")
        if bytes_read.value != self._chunk_bytes:
            raise OSError(
                f"short unbuffered read at offset {offset}: "
                f"{bytes_read.value}/{self._chunk_bytes} bytes"
            )
        first = ctypes.c_ubyte.from_address(self._buffer_address).value
        last = ctypes.c_ubyte.from_address(
            self._buffer_address + self._chunk_bytes - 1
        ).value
        return first ^ last

    def close(self) -> None:
        if self._allocation is not None:
            self._kernel32.VirtualFree(self._allocation, 0, self._MEM_RELEASE)
            self._allocation = None
            self._buffer_address = None
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _open_read_handle(
    target: _TargetSnapshot,
    *,
    chunk_bytes: int,
    alignment_bytes: int,
    io_mode: str,
) -> _RandomReadHandle:
    if io_mode == "buffered":
        return _BufferedReadHandle(target.path, chunk_bytes, alignment_bytes)
    if io_mode == "windows-unbuffered":
        return _WindowsUnbufferedReadHandle(target.path, chunk_bytes, alignment_bytes)
    raise ValueError(f"unsupported I/O mode: {io_mode}")


def _validate_parameters(
    *,
    chunk_bytes: int,
    operations_per_target: int,
    warmup_operations_per_target: int,
    alignment_bytes: int,
) -> None:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if operations_per_target <= 0:
        raise ValueError("operations_per_target must be positive")
    if warmup_operations_per_target < 0:
        raise ValueError("warmup_operations_per_target cannot be negative")
    if alignment_bytes <= 0:
        raise ValueError("alignment_bytes must be positive")


def _capture_targets(paths: list[str | os.PathLike[str]]) -> list[_TargetSnapshot]:
    if not paths:
        raise ValueError("at least one target file is required")
    targets = [_TargetSnapshot.capture(path) for path in paths]
    normalized = [os.path.normcase(str(target.path)) for target in targets]
    if len(normalized) != len(set(normalized)):
        raise ValueError("the same target file was provided more than once")
    return targets


def _run_reads(
    handle: _RandomReadHandle,
    target: _TargetSnapshot,
    rng: random.Random,
    *,
    chunk_bytes: int,
    operations: int,
) -> tuple[list[int], int, int]:
    latencies_ns: list[int] = []
    sink = 0
    started_ns = time.perf_counter_ns()
    for _ in range(operations):
        offset = _random_aligned_offset(
            rng,
            file_size=target.size_bytes,
            chunk_bytes=chunk_bytes,
            alignment_bytes=handle.alignment_bytes,
        )
        operation_started_ns = time.perf_counter_ns()
        sink ^= handle.read_at(offset)
        latencies_ns.append(time.perf_counter_ns() - operation_started_ns)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return latencies_ns, elapsed_ns, sink


def benchmark_storage(
    paths: list[str | os.PathLike[str]],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    operations_per_target: int = 64,
    warmup_operations_per_target: int = 4,
    seed: int = 20260810,
    io_mode: str = "buffered",
    alignment_bytes: int = DEFAULT_ALIGNMENT_BYTES,
) -> dict[str, Any]:
    """Measure random fixed-size reads without modifying the target files."""
    _validate_parameters(
        chunk_bytes=chunk_bytes,
        operations_per_target=operations_per_target,
        warmup_operations_per_target=warmup_operations_per_target,
        alignment_bytes=alignment_bytes,
    )
    targets = _capture_targets(paths)
    for target in targets:
        if target.size_bytes < chunk_bytes:
            raise ValueError(
                f"target {target.path} is smaller than one {chunk_bytes}-byte chunk"
            )

    started_at = datetime.now(UTC)
    rng = random.Random(seed)
    target_results: list[dict[str, Any]] = []
    all_latencies_ns: list[int] = []
    total_elapsed_ns = 0
    total_warmup_elapsed_ns = 0
    combined_sink = 0

    for target in targets:
        handle = _open_read_handle(
            target,
            chunk_bytes=chunk_bytes,
            alignment_bytes=alignment_bytes,
            io_mode=io_mode,
        )
        try:
            _, warmup_elapsed_ns, warmup_sink = _run_reads(
                handle,
                target,
                rng,
                chunk_bytes=chunk_bytes,
                operations=warmup_operations_per_target,
            )
            latencies_ns, elapsed_ns, measured_sink = _run_reads(
                handle,
                target,
                rng,
                chunk_bytes=chunk_bytes,
                operations=operations_per_target,
            )
        finally:
            handle.close()

        target.assert_unchanged()
        measured_bytes = operations_per_target * chunk_bytes
        throughput_mib_s = measured_bytes / MIB / (elapsed_ns / 1_000_000_000)
        target_results.append(
            {
                "path": str(target.path),
                "volume": target.path.anchor,
                "size_bytes": target.size_bytes,
                "modified_time_ns": target.modified_time_ns,
                "effective_offset_alignment_bytes": handle.alignment_bytes,
                "operations": operations_per_target,
                "bytes_read": measured_bytes,
                "elapsed_seconds": elapsed_ns / 1_000_000_000,
                "throughput_mib_s": throughput_mib_s,
                "latency_ms": _latency_summary(latencies_ns),
                "warmup": {
                    "operations": warmup_operations_per_target,
                    "bytes_read": warmup_operations_per_target * chunk_bytes,
                    "elapsed_seconds": warmup_elapsed_ns / 1_000_000_000,
                },
                "read_sink_uint8": measured_sink ^ warmup_sink,
                "unchanged_after_benchmark": True,
            }
        )
        all_latencies_ns.extend(latencies_ns)
        total_elapsed_ns += elapsed_ns
        total_warmup_elapsed_ns += warmup_elapsed_ns
        combined_sink ^= measured_sink ^ warmup_sink

    total_operations = operations_per_target * len(targets)
    total_bytes = total_operations * chunk_bytes
    os_cache_bypassed = io_mode == "windows-unbuffered"
    if os_cache_bypassed:
        cache_note = (
            "Windows FILE_FLAG_NO_BUFFERING bypasses the normal system file cache, "
            "but device, controller, and SSD caches may still affect results. Cold "
            "cache state is not guaranteed."
        )
    else:
        cache_note = (
            "Normal reads use the operating-system page cache. Warmup is reported, "
            "but cache state is neither flushed nor controlled; repeated or already "
            "cached data can raise measured throughput."
        )

    completed_at = datetime.now(UTC)
    return {
        "schema_version": 1,
        "benchmark_type": "microbenchmark",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "parameters": {
            "chunk_bytes": chunk_bytes,
            "operations_per_target": operations_per_target,
            "warmup_operations_per_target": warmup_operations_per_target,
            "seed": seed,
            "io_mode": io_mode,
            "requested_offset_alignment_bytes": alignment_bytes,
            "access_pattern": "uniform_random_aligned_offsets_with_replacement",
            "target_order_matters_for_seed": True,
        },
        "cache_policy": {
            "os_page_cache_bypassed": os_cache_bypassed,
            "cold_cache_guaranteed": False,
            "limitations": cache_note,
        },
        "safety": {
            "target_access": "read_only",
            "target_writes": 0,
            "metadata_checked_after_run": ["size_bytes", "modified_time_ns"],
            "all_targets_unchanged": True,
        },
        "summary": {
            "targets": len(targets),
            "operations": total_operations,
            "bytes_read": total_bytes,
            "elapsed_seconds": total_elapsed_ns / 1_000_000_000,
            "throughput_mib_s": total_bytes / MIB / (total_elapsed_ns / 1_000_000_000),
            "latency_ms": _latency_summary(all_latencies_ns),
            "warmup": {
                "operations": warmup_operations_per_target * len(targets),
                "bytes_read": warmup_operations_per_target * len(targets) * chunk_bytes,
                "elapsed_seconds": total_warmup_elapsed_ns / 1_000_000_000,
            },
            "read_sink_uint8": combined_sink,
            "all_targets_unchanged": True,
        },
        "targets": target_results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moevm.storage_benchmark",
        description=(
            "Read-only random-chunk storage microbenchmark. Results are JSON and do "
            "not represent end-to-end MoE inference."
        ),
    )
    parser.add_argument("files", nargs="+", help="one or more large target files")
    parser.add_argument(
        "--chunk-size",
        type=parse_size,
        default=DEFAULT_CHUNK_BYTES,
        help="bytes per read (default: 12MiB)",
    )
    parser.add_argument(
        "--operations",
        "--operations-per-target",
        dest="operations_per_target",
        type=int,
        default=64,
        help="measured reads per target (default: 64)",
    )
    parser.add_argument(
        "--warmup",
        "--warmup-operations-per-target",
        dest="warmup_operations_per_target",
        type=int,
        default=4,
        help="unmeasured warmup reads per target (default: 4)",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--io-mode",
        choices=("buffered", "windows-unbuffered"),
        default="buffered",
    )
    parser.add_argument(
        "--offset-alignment",
        type=parse_size,
        default=DEFAULT_ALIGNMENT_BYTES,
        help="random-offset alignment (default: 4096)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="create this JSON file; existing files are never overwritten",
    )
    return parser


def _validate_output_path(output: Path, target_paths: list[str]) -> Path:
    resolved_output = output.expanduser().resolve(strict=False)
    normalized_output = os.path.normcase(str(resolved_output))
    for target_path in target_paths:
        resolved_target = Path(target_path).expanduser().resolve(strict=True)
        if normalized_output == os.path.normcase(str(resolved_target)):
            raise ValueError("the JSON output path cannot be a benchmark target")
    if resolved_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output file: {resolved_output}"
        )
    return resolved_output


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output = _validate_output_path(args.output, args.files) if args.output else None
        result = benchmark_storage(
            args.files,
            chunk_bytes=args.chunk_size,
            operations_per_target=args.operations_per_target,
            warmup_operations_per_target=args.warmup_operations_per_target,
            seed=args.seed,
            io_mode=args.io_mode,
            alignment_bytes=args.offset_alignment,
        )
        payload = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
        if output is None:
            sys.stdout.write(payload)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
            print(f"Wrote {output}", file=sys.stderr)
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
