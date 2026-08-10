"""Reproducible, bounded host-RAM to CUDA-device transfer benchmark."""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
GIB = 1024**3
DEFAULT_CHUNK_BYTES = 12 * MIB
MAX_CHUNK_BYTES = 256 * MIB
DEFAULT_OPERATIONS = 64
DEFAULT_WARMUP_OPERATIONS = 8
DEFAULT_ASYNC_DEPTH = 8
MAX_ASYNC_DEPTH = 64
# Each measured operation owns two CUDA timing events. Keep the hard ceiling
# conservative so a typo cannot create an unbounded event queue.
MAX_OPERATIONS = 4096
DEVICE_SAFETY_RESERVE_BYTES = 64 * MIB


@dataclass(frozen=True)
class _CaseSpec:
    host_memory: str
    transfer_mode: str
    non_blocking: bool

    @property
    def name(self) -> str:
        return f"{self.host_memory}-{self.transfer_mode}"


def parse_size(value: str) -> int:
    """Parse an integer byte count or a common binary/decimal size suffix."""
    text = value.strip().lower()
    suffixes = {
        "kib": 1024,
        "mib": MIB,
        "gib": GIB,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "b": 1,
    }
    number = text
    multiplier = 1
    for suffix in sorted(suffixes, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            multiplier = suffixes[suffix]
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
        raise ValueError("cannot summarize an empty latency sample")
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


def _latency_summary(values_ms: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values_ms),
        "min": min(values_ms),
        "mean": sum(values_ms) / len(values_ms),
        "p50": _percentile(values_ms, 0.50),
        "p95": _percentile(values_ms, 0.95),
        "p99": _percentile(values_ms, 0.99),
        "max": max(values_ms),
    }


def _verification_offsets(chunk_bytes: int) -> tuple[int, ...]:
    """Return bounded positions spanning the copied payload."""
    return tuple(sorted({0, chunk_bytes // 4, chunk_bytes // 2, chunk_bytes - 1}))


def _case_specs(mode: str) -> list[_CaseSpec]:
    sync_cases = [
        _CaseSpec("pageable", "sync", False),
        _CaseSpec("pinned", "sync", False),
    ]
    async_cases = [_CaseSpec("pinned", "async", True)]
    if mode == "sync":
        return sync_cases
    if mode == "async":
        return async_cases
    if mode == "both":
        return sync_cases + async_cases
    raise ValueError(f"unsupported transfer mode: {mode}")


def _validate_parameters(
    *,
    chunk_bytes: int,
    operations: int,
    warmup_operations: int,
    async_depth: int,
    device_index: int,
    mode: str,
) -> None:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if chunk_bytes > MAX_CHUNK_BYTES:
        raise ValueError(
            f"chunk_bytes exceeds the {MAX_CHUNK_BYTES}-byte safety limit: "
            f"{chunk_bytes}"
        )
    if not 0 < operations <= MAX_OPERATIONS:
        raise ValueError(f"operations must be between 1 and {MAX_OPERATIONS}")
    if not 0 <= warmup_operations <= MAX_OPERATIONS:
        raise ValueError(f"warmup_operations must be between 0 and {MAX_OPERATIONS}")
    if not 0 < async_depth <= MAX_ASYNC_DEPTH:
        raise ValueError(f"async_depth must be between 1 and {MAX_ASYNC_DEPTH}")
    if device_index < 0:
        raise ValueError("device_index cannot be negative")
    _case_specs(mode)


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a CUDA-enabled PyTorch build in "
            "the benchmark environment."
        ) from exc
    return torch


def _cuda_device(torch: Any, device_index: int) -> Any:
    if not torch.cuda.is_available():
        build = getattr(getattr(torch, "version", None), "cuda", None)
        detail = (
            "this appears to be a CPU-only PyTorch build"
            if build is None
            else "the CUDA runtime or GPU is unavailable"
        )
        raise RuntimeError(f"CUDA is not available; {detail}")
    device_count = int(torch.cuda.device_count())
    if device_index >= device_count:
        raise ValueError(
            f"CUDA device index {device_index} is out of range; "
            f"found {device_count} device(s)"
        )
    return torch.device(f"cuda:{device_index}")


def _device_metadata(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "index": int(device.index),
        "name": properties.name,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "multiprocessor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "free_memory_bytes_before": int(free_bytes),
        "mem_get_info_total_bytes": int(total_bytes),
    }


def _software_metadata(torch: Any) -> dict[str, Any]:
    cudnn = None
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        cudnn = torch.backends.cudnn.version()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": cudnn,
    }


def _warm_up(
    torch: Any,
    *,
    source: Any,
    destination: Any,
    stream: Any,
    operations: int,
    non_blocking: bool,
) -> float:
    stream.synchronize()
    started_ns = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        for _ in range(operations):
            destination.copy_(source, non_blocking=non_blocking)
    stream.synchronize()
    return (time.perf_counter_ns() - started_ns) / 1_000_000_000


def _measure_sync(
    torch: Any,
    *,
    source: Any,
    destination: Any,
    stream: Any,
    operations: int,
) -> tuple[list[float], list[float], list[float], float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(operations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(operations)]
    host_call_ms: list[float] = []
    completion_ms: list[float] = []

    stream.synchronize()
    run_started_ns = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        for index in range(operations):
            operation_started_ns = time.perf_counter_ns()
            starts[index].record(stream)
            call_started_ns = time.perf_counter_ns()
            destination.copy_(source, non_blocking=False)
            host_call_ms.append((time.perf_counter_ns() - call_started_ns) / 1_000_000)
            ends[index].record(stream)
            stream.synchronize()
            completion_ms.append(
                (time.perf_counter_ns() - operation_started_ns) / 1_000_000
            )
    elapsed_seconds = (time.perf_counter_ns() - run_started_ns) / 1_000_000_000
    device_event_ms = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)
    ]
    return device_event_ms, host_call_ms, completion_ms, elapsed_seconds


def _measure_async(
    torch: Any,
    *,
    source: Any,
    destination: Any,
    stream: Any,
    operations: int,
    async_depth: int,
) -> tuple[list[float], list[float], list[float], float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(operations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(operations)]
    host_call_ms: list[float] = []
    batch_completion_ms: list[float] = []

    stream.synchronize()
    run_started_ns = time.perf_counter_ns()
    for batch_start in range(0, operations, async_depth):
        batch_stop = min(batch_start + async_depth, operations)
        batch_started_ns = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            for index in range(batch_start, batch_stop):
                starts[index].record(stream)
                call_started_ns = time.perf_counter_ns()
                destination.copy_(source, non_blocking=True)
                host_call_ms.append(
                    (time.perf_counter_ns() - call_started_ns) / 1_000_000
                )
                ends[index].record(stream)
        stream.synchronize()
        batch_completion_ms.append(
            (time.perf_counter_ns() - batch_started_ns) / 1_000_000
        )
    elapsed_seconds = (time.perf_counter_ns() - run_started_ns) / 1_000_000_000
    device_event_ms = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)
    ]
    return device_event_ms, host_call_ms, batch_completion_ms, elapsed_seconds


def _run_case(
    torch: Any,
    *,
    device: Any,
    spec: _CaseSpec,
    chunk_bytes: int,
    operations: int,
    warmup_operations: int,
    async_depth: int,
) -> dict[str, Any]:
    pin_memory = spec.host_memory == "pinned"
    source = torch.empty(
        chunk_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=pin_memory,
    )
    source.fill_(0xA5)
    destination = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)

    try:
        warmup_elapsed = _warm_up(
            torch,
            source=source,
            destination=destination,
            stream=stream,
            operations=warmup_operations,
            non_blocking=spec.non_blocking,
        )
        if spec.transfer_mode == "sync":
            event_ms, host_call_ms, completion_ms, elapsed_seconds = _measure_sync(
                torch,
                source=source,
                destination=destination,
                stream=stream,
                operations=operations,
            )
            completion_name = "host_completion"
        else:
            event_ms, host_call_ms, completion_ms, elapsed_seconds = _measure_async(
                torch,
                source=source,
                destination=destination,
                stream=stream,
                operations=operations,
                async_depth=async_depth,
            )
            completion_name = "async_batch_completion"

        stream.synchronize()
        verification_offsets = _verification_offsets(chunk_bytes)
        verification_values = tuple(
            int(destination[offset].item()) for offset in verification_offsets
        )
        if any(value != 0xA5 for value in verification_values):
            raise RuntimeError(
                f"transfer verification failed for {spec.name}: "
                f"expected 165 at {verification_offsets}, found "
                f"{verification_values}"
            )
        transferred_bytes = operations * chunk_bytes
        bytes_per_second = transferred_bytes / elapsed_seconds
        return {
            "name": spec.name,
            "host_memory": spec.host_memory,
            "source_is_pinned": bool(source.is_pinned()),
            "transfer_mode": spec.transfer_mode,
            "non_blocking": spec.non_blocking,
            "async_depth": async_depth if spec.transfer_mode == "async" else 1,
            "operations": operations,
            "bytes_transferred": transferred_bytes,
            "elapsed_seconds": elapsed_seconds,
            "throughput": {
                "bytes_per_second": bytes_per_second,
                "gb_per_second": bytes_per_second / 1_000_000_000,
                "gib_per_second": bytes_per_second / GIB,
                "mib_per_second": bytes_per_second / MIB,
                "basis": (
                    "payload bytes divided by synchronized host wall time; "
                    "includes the requested synchronization policy"
                ),
            },
            "latency_ms": {
                "cuda_event": _latency_summary(event_ms),
                "host_submission": _latency_summary(host_call_ms),
                completion_name: _latency_summary(completion_ms),
            },
            "warmup": {
                "operations": warmup_operations,
                "bytes_transferred": warmup_operations * chunk_bytes,
                "elapsed_seconds": warmup_elapsed,
            },
            "verification": {
                "expected_uint8": 0xA5,
                "device_sample_offsets": list(verification_offsets),
                "device_sample_values_uint8": list(verification_values),
                "passed": True,
            },
        }
    finally:
        del stream
        del destination
        del source
        torch.cuda.empty_cache()


def benchmark_cuda_transfers(
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    operations: int = DEFAULT_OPERATIONS,
    warmup_operations: int = DEFAULT_WARMUP_OPERATIONS,
    async_depth: int = DEFAULT_ASYNC_DEPTH,
    device_index: int = 0,
    mode: str = "both",
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Measure bounded pageable/pinned host-to-device CUDA transfers."""
    _validate_parameters(
        chunk_bytes=chunk_bytes,
        operations=operations,
        warmup_operations=warmup_operations,
        async_depth=async_depth,
        device_index=device_index,
        mode=mode,
    )
    torch = torch_module if torch_module is not None else _load_torch()
    device = _cuda_device(torch, device_index)
    torch.cuda.set_device(device)
    device_metadata = _device_metadata(torch, device)
    free_bytes = int(device_metadata["free_memory_bytes_before"])
    if chunk_bytes + DEVICE_SAFETY_RESERVE_BYTES > free_bytes:
        raise RuntimeError(
            "insufficient free VRAM for the bounded benchmark allocation: "
            f"need {chunk_bytes + DEVICE_SAFETY_RESERVE_BYTES} bytes including "
            f"reserve, found {free_bytes}"
        )

    started_at = datetime.now(UTC)
    results = [
        _run_case(
            torch,
            device=device,
            spec=spec,
            chunk_bytes=chunk_bytes,
            operations=operations,
            warmup_operations=warmup_operations,
            async_depth=async_depth,
        )
        for spec in _case_specs(mode)
    ]
    free_after, _ = torch.cuda.mem_get_info(device)
    device_metadata["free_memory_bytes_after"] = int(free_after)

    baseline = next(
        (result for result in results if result["name"] == "pageable-sync"), None
    )
    baseline_rate = (
        float(baseline["throughput"]["bytes_per_second"]) if baseline else None
    )
    for result in results:
        result["throughput_vs_pageable_sync"] = (
            float(result["throughput"]["bytes_per_second"]) / baseline_rate
            if baseline_rate
            else None
        )

    completed_at = datetime.now(UTC)
    return {
        "schema_version": 1,
        "benchmark_type": "host_to_device_cuda_transfer_microbenchmark",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "software": _software_metadata(torch),
        "device": device_metadata,
        "parameters": {
            "chunk_bytes": chunk_bytes,
            "operations": operations,
            "warmup_operations": warmup_operations,
            "async_depth": async_depth,
            "device_index": device_index,
            "mode": mode,
            "source_pattern_uint8": 0xA5,
        },
        "safety": {
            "maximum_chunk_bytes": MAX_CHUNK_BYTES,
            "maximum_measured_operations_per_case": MAX_OPERATIONS,
            "device_safety_reserve_bytes": DEVICE_SAFETY_RESERVE_BYTES,
            "peak_benchmark_host_payload_bytes": chunk_bytes,
            "peak_benchmark_device_payload_bytes": chunk_bytes,
            "cases_run_sequentially": True,
            "model_weights_loaded": False,
        },
        "methodology": {
            "stream": "dedicated CUDA stream per case",
            "latency": (
                "CUDA-event transfer duration plus host submission timing; sync "
                "cases also report per-transfer completion latency, while async "
                "reports synchronized batch completion latency"
            ),
            "async_scope": (
                "async is measured only from pinned host memory because pageable "
                "memory cannot provide reliable asynchronous H2D DMA"
            ),
            "limitations": (
                "isolated copies only; results do not include model execution, "
                "PCIe contention, NUMA placement, or end-to-end inference"
            ),
        },
        "results": results,
    }


def _validate_output_path(output: Path) -> Path:
    resolved = output.expanduser().resolve(strict=False)
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing output file: {resolved}")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moevm.cuda_transfer_benchmark",
        description=(
            "Bounded PyTorch/CUDA host-RAM to VRAM transfer microbenchmark. "
            "Prints JSON and does not load model weights."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=parse_size,
        default=DEFAULT_CHUNK_BYTES,
        help="bytes copied per transfer (default: 12MiB; safety limit: 256MiB)",
    )
    parser.add_argument(
        "--operations",
        type=int,
        default=DEFAULT_OPERATIONS,
        help="measured transfers per case (default: 64)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_OPERATIONS,
        help="unmeasured transfers per case (default: 8)",
    )
    parser.add_argument(
        "--mode",
        choices=("sync", "async", "both"),
        default="both",
        help="sync runs pageable and pinned; async runs pinned only (default: both)",
    )
    parser.add_argument(
        "--async-depth",
        type=int,
        default=DEFAULT_ASYNC_DEPTH,
        help="copies submitted before each async stream synchronization (default: 8)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--output",
        type=Path,
        help="create this JSON file; existing files are never overwritten",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output = _validate_output_path(args.output) if args.output else None
        report = benchmark_cuda_transfers(
            chunk_bytes=args.chunk_size,
            operations=args.operations,
            warmup_operations=args.warmup,
            async_depth=args.async_depth,
            device_index=args.device,
            mode=args.mode,
        )
        payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
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
