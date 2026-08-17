#!/usr/bin/env python3
"""Run one controlled offline OLMoE paged-runtime cold/warm smoke benchmark."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moevm.olmoe_assets import (
    PINNED_MODEL_ID,
    PINNED_REVISION,
    PINNED_SHARD_SHA256,
    PINNED_SHARD_SIZES,
)
from moevm.pipeline_profile import (
    load_pipeline_profile,
    result_binding,
    validate_profile_binding,
)
from moevm.timeline_metrics import CudaInterval, summarize_cuda_timeline

_HASH_BLOCK_BYTES = 16 * 1024 * 1024
_VRAM_SAFETY_MARGIN_BYTES = int(1.25 * 1024**3)
_DEFAULT_PROMPT = "Explain sparse mixture-of-experts in one short sentence."
_INTEGER_METRICS = (
    "requests",
    "hits",
    "misses",
    "evictions",
    "storage_bytes",
    "host_to_device_bytes",
)
_PIPELINE_INTEGER_METRICS = (
    "coalesced_requests",
    "storage_loads",
    "transfer_loads",
    "admission_rejections",
    "storage_failures",
    "transfer_failures",
    "staging_waits",
    "proactive_h2d_slot_declines",
    "adaptive_async_forwards",
    "adaptive_sync_forwards",
    "adaptive_async_experts",
    "adaptive_sync_experts",
)
_PIPELINE_HIGH_WATER_METRICS = (
    "pending_loads_peak",
    "peak_staging_in_use",
)
_TIME_METRICS = (
    "storage_seconds",
    "transfer_seconds",
    "forward_seconds",
)
_PIPELINE_TIME_METRICS = (
    "storage_queue_seconds",
    "reader_queue_wait_seconds",
    "staging_wait_seconds",
    "demand_wait_seconds",
)
# The benchmark report itself remains schema v1.  Only the opt-in nested CUDA
# telemetry contract is v2, because v1 did not prove captured H2D coverage.
_CUDA_TIMELINE_SCHEMA_VERSION = 2
_LEGACY_CUDA_TIMELINE_SCHEMA_VERSION = 1
_CUDA_TIMELINE_METHOD = "cuda_events_v1"
_CUDA_TIMELINE_SCOPE = "paged_expert_h2d_vs_expert_compute"
_CUDA_TIMELINE_UNIT = "milliseconds"
_CUDA_TIMELINE_AGGREGATION = (
    "Summed per-model-call CUDA-event lane summaries; timestamps from different "
    "model calls are not unioned."
)


def _bounded_integer(name: str, minimum: int, maximum: int):
    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        required=True,
        help=f"Local snapshot directory named {PINNED_REVISION}; no download occurs.",
    )
    parser.add_argument("--output", required=True, help="Create-only JSON result path.")
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help=(
            "allow best-effort Git provenance for an interactive local demo; "
            "the result is explicitly not publishable benchmark evidence"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
    )
    parser.add_argument(
        "--workload-file",
        help="Optional v1 workload collection; selects --workload-id from it.",
    )
    parser.add_argument("--workload-id", default="single-smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy", choices=("lru", "hybrid"), default="lru")
    parser.add_argument(
        "--pipeline",
        choices=("sync", "async", "adaptive", "auto"),
        default="sync",
        help=(
            "Expert data path. Async uses one bounded storage worker and a "
            "dedicated CUDA H2D stream; adaptive uses it for routed-layer "
            "calls that start with free slots and multiple misses; auto uses "
            "a measured hardware-bound profile; sync preserves the v0.3.0 "
            "behavior."
        ),
    )
    parser.add_argument(
        "--pipeline-profile",
        help="Required for --pipeline auto; measured v1 selection profile.",
    )
    parser.add_argument(
        "--cuda-overlap-telemetry",
        action="store_true",
        help=(
            "Record opt-in same-device CUDA-event timelines for paged-expert "
            "H2D copies and expert compute. This is instrumentation, not a "
            "physical NVMe or general-speedup measurement."
        ),
    )
    parser.add_argument(
        "--slots-per-layer",
        type=_bounded_integer("slots-per-layer", 1, 48),
        default=32,
        help="Independent GPU slots in each of 16 layers (32 = 6 GiB).",
    )
    parser.add_argument(
        "--hotset-json",
        help="Required for hybrid: explicit v1 model/revision-bound per-layer hotsets.",
    )
    parser.add_argument(
        "--staging-slots",
        type=_bounded_integer("staging-slots", 1, 4),
        default=1,
    )
    parser.add_argument(
        "--max-input-tokens",
        type=_bounded_integer("max-input-tokens", 1, 256),
        default=64,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_bounded_integer("max-new-tokens", 1, 64),
        default=2,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--reference-metadata",
        help="Optional v1 pinned greedy-baseline metadata used as a correctness gate.",
    )
    parser.add_argument(
        "--teacher-force-reference",
        action="store_true",
        help=(
            "Feed the pinned reference continuation while recording greedy "
            "predictions. Requires --reference-metadata and keeps the default "
            "autoregressive exact-match gate unchanged."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("prompt cannot be empty")
    if args.prompt is not None and args.workload_file:
        raise ValueError("use either --prompt or --workload-file, not both")
    if not args.workload_id.strip():
        raise ValueError("workload-id cannot be empty")
    if not re_fullmatch_cuda(args.device):
        raise ValueError("device must be 'cuda' or 'cuda:<index>'")
    if args.policy == "hybrid" and not args.hotset_json:
        raise ValueError("hybrid policy requires --hotset-json")
    if args.policy == "lru" and args.hotset_json:
        raise ValueError("--hotset-json is only valid with --policy hybrid")
    if args.teacher_force_reference and not args.reference_metadata:
        raise ValueError("--teacher-force-reference requires --reference-metadata")
    if args.pipeline == "auto" and not args.pipeline_profile:
        raise ValueError("--pipeline auto requires --pipeline-profile")
    if args.pipeline == "auto" and args.demo_mode:
        raise ValueError("--pipeline auto is not available in --demo-mode")
    if args.pipeline != "auto" and args.pipeline_profile:
        raise ValueError("--pipeline-profile is only valid with --pipeline auto")
    if args.pipeline in ("async", "adaptive", "auto") and args.staging_slots < 2:
        raise ValueError("async-capable pipeline requires at least two staging slots")
    output_path = Path(args.output).expanduser()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")


def _resolve_prompt(args: argparse.Namespace) -> str:
    if not args.workload_file:
        return args.prompt if args.prompt is not None else _DEFAULT_PROMPT
    workload_path = Path(args.workload_file)
    payload = json.loads(workload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("workload file must be a schema_version 1 object")
    workloads = payload.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("workload file must contain a workloads array")
    matching = [
        item
        for item in workloads
        if isinstance(item, dict) and item.get("id") == args.workload_id
    ]
    if len(matching) != 1 or not isinstance(matching[0].get("prompt"), str):
        raise ValueError(
            f"workload-id must match exactly one prompt: {args.workload_id}"
        )
    return matching[0]["prompt"]


def re_fullmatch_cuda(device: str) -> bool:
    if device == "cuda":
        return True
    if not device.startswith("cuda:"):
        return False
    index = device.removeprefix("cuda:")
    return index.isdigit() and int(index) >= 0


def _validate_snapshot(snapshot: Path) -> Path:
    resolved = snapshot.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {resolved}")
    if resolved.name != PINNED_REVISION:
        raise ValueError(
            f"snapshot directory must be the pinned revision {PINNED_REVISION}"
        )
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for filename in required:
        if not (resolved / filename).is_file():
            raise FileNotFoundError(f"snapshot file not found: {resolved / filename}")
    return resolved


def _sha256_stream(path: Path, *, block_bytes: int = _HASH_BLOCK_BYTES) -> str:
    if block_bytes <= 0:
        raise ValueError("hash block size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_pinned_shards(
    snapshot: Path,
    *,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, dict[str, int | str]]:
    expected_hashes = expected_hashes or PINNED_SHARD_SHA256
    verified: dict[str, dict[str, int | str]] = {}
    for filename, expected in expected_hashes.items():
        shard_path = snapshot / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard not found: {shard_path}")
        actual = _sha256_stream(shard_path)
        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch for {filename}: {actual} != {expected}"
            )
        verified[filename] = {
            "sha256": actual,
            "size_bytes": shard_path.stat().st_size,
        }
    return verified


def _close_store_and_verify_pinned_shards(
    store: Any,
    snapshot: Path,
) -> dict[str, dict[str, int | str]]:
    """Release live mappings before the independent checkpoint integrity pass."""
    store.close()
    return _verify_pinned_shards(snapshot)


def _validate_pinned_shard_files(snapshot: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for filename, expected_size in PINNED_SHARD_SIZES.items():
        shard_path = snapshot / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard not found: {shard_path}")
        actual_size = shard_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"checkpoint size mismatch for {filename}: "
                f"{actual_size} != {expected_size}"
            )
        sizes[filename] = actual_size
    return sizes


def _load_hotsets(
    path: Path,
    *,
    layers: tuple[int, ...],
    experts_per_layer: int,
    slots_per_layer: int,
) -> tuple[dict[int, tuple[int, ...]], str]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict):
        raise ValueError("hotset JSON must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("hotset JSON schema_version must be 1")
    if payload.get("model_id") != PINNED_MODEL_ID:
        raise ValueError("hotset JSON model_id does not match the pinned model")
    if payload.get("revision") != PINNED_REVISION:
        raise ValueError("hotset JSON revision does not match the pinned revision")
    hotsets = payload.get("hotsets")
    if not isinstance(hotsets, dict):
        raise ValueError("hotset JSON must contain a hotsets object")
    expected_layer_keys = {str(layer) for layer in layers}
    if set(hotsets) != expected_layer_keys:
        raise ValueError("hotset layers must exactly match the checkpoint layers")

    normalized: dict[int, tuple[int, ...]] = {}
    for layer in layers:
        experts = hotsets[str(layer)]
        if not isinstance(experts, list):
            raise ValueError(f"hotset layer {layer} must be an array")
        if not experts:
            raise ValueError(f"hotset layer {layer} cannot be empty")
        if len(experts) >= slots_per_layer:
            raise ValueError(
                f"hotset layer {layer} must leave at least one dynamic LRU slot"
            )
        if any(
            isinstance(expert, bool) or not isinstance(expert, int)
            for expert in experts
        ):
            raise ValueError(f"hotset layer {layer} must contain integers")
        if len(set(experts)) != len(experts):
            raise ValueError(f"hotset layer {layer} contains duplicate experts")
        if any(not 0 <= expert < experts_per_layer for expert in experts):
            raise ValueError(f"hotset layer {layer} contains an out-of-range expert")
        normalized[layer] = tuple(experts)
    return normalized, hashlib.sha256(raw_bytes).hexdigest()


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_reference_metadata(
    path: Path,
    *,
    workload_id: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("reference metadata must be a schema_version 1 object")
    model = payload.get("model")
    workload = payload.get("workload")
    generation = payload.get("generation")
    if not all(isinstance(item, dict) for item in (model, workload, generation)):
        raise ValueError(
            "reference metadata model/workload/generation objects are required"
        )
    if model.get("id") != PINNED_MODEL_ID:
        raise ValueError("reference metadata model id does not match")
    if (
        model.get("requested_revision") != PINNED_REVISION
        or model.get("resolved_revision") != PINNED_REVISION
    ):
        raise ValueError("reference metadata revision does not match")
    if workload.get("id") != workload_id:
        raise ValueError("reference metadata workload id does not match")
    if workload.get("prompt_sha256") != _prompt_sha256(prompt):
        raise ValueError("reference metadata prompt SHA-256 does not match")
    temperature = generation.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("reference metadata temperature must be numeric")
    if float(temperature) != 0.0:
        raise ValueError("reference metadata must use greedy temperature 0")
    token_ids = generation.get("generated_token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(token, bool) or not isinstance(token, int) for token in token_ids
        )
    ):
        raise ValueError(
            "reference generated_token_ids must be a non-empty integer array"
        )
    token_prefix = token_ids[:max_new_tokens]
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "generated_token_ids": token_prefix,
        "source_generated_token_count": len(token_ids),
        "temperature": 0.0,
    }


def _metrics_payload(metrics: Any) -> dict[str, int | float]:
    payload: dict[str, int | float] = {
        name: int(getattr(metrics, name)) for name in _INTEGER_METRICS
    }
    payload.update({name: float(getattr(metrics, name)) for name in _TIME_METRICS})
    payload.update(
        {
            name: int(getattr(metrics, name))
            for name in (*_PIPELINE_INTEGER_METRICS, *_PIPELINE_HIGH_WATER_METRICS)
            if hasattr(metrics, name)
        }
    )
    payload.update(
        {
            name: float(getattr(metrics, name))
            for name in _PIPELINE_TIME_METRICS
            if hasattr(metrics, name)
        }
    )
    requests = int(payload["requests"])
    payload["hit_rate"] = 0.0 if requests == 0 else int(payload["hits"]) / requests
    return payload


def _metrics_delta(after: Any, before: Any) -> dict[str, int | float]:
    payload: dict[str, int | float] = {
        name: int(getattr(after, name)) - int(getattr(before, name))
        for name in _INTEGER_METRICS
    }
    payload.update(
        {
            name: float(getattr(after, name)) - float(getattr(before, name))
            for name in _TIME_METRICS
        }
    )
    payload.update(
        {
            name: int(getattr(after, name)) - int(getattr(before, name))
            for name in _PIPELINE_INTEGER_METRICS
            if hasattr(after, name) and hasattr(before, name)
        }
    )
    payload.update(
        {
            name: float(getattr(after, name)) - float(getattr(before, name))
            for name in _PIPELINE_TIME_METRICS
            if hasattr(after, name) and hasattr(before, name)
        }
    )
    requests = int(payload["requests"])
    payload["hit_rate"] = 0.0 if requests == 0 else int(payload["hits"]) / requests
    return payload


def _validate_metric_delta(
    payload: dict[str, int | float],
    *,
    expert_bytes: int,
) -> None:
    if int(payload["requests"]) != int(payload["hits"]) + int(payload["misses"]):
        raise RuntimeError("cache metric invariant failed: requests != hits + misses")
    storage_loads = int(payload.get("storage_loads", payload["misses"]))
    transfer_loads = int(payload.get("transfer_loads", payload["misses"]))
    if storage_loads > int(payload["misses"]):
        raise RuntimeError("cache metric invariant failed: storage loads exceed misses")
    if transfer_loads > storage_loads:
        raise RuntimeError(
            "cache metric invariant failed: transfers exceed storage loads"
        )
    expected_storage_bytes = storage_loads * expert_bytes
    if int(payload["storage_bytes"]) != expected_storage_bytes:
        raise RuntimeError("cache metric invariant failed for logical storage bytes")
    expected_transfer_bytes = transfer_loads * expert_bytes
    if int(payload["host_to_device_bytes"]) != expected_transfer_bytes:
        raise RuntimeError("cache metric invariant failed for logical H2D bytes")
    if any(float(value) < 0.0 for name, value in payload.items() if name != "hit_rate"):
        raise RuntimeError("cache metric delta cannot be negative")


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _process_memory() -> dict[str, int | None]:
    if os.name == "nt":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if success:
            return {
                "rss_bytes": int(counters.WorkingSetSize),
                "peak_rss_bytes": int(counters.PeakWorkingSetSize),
            }
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        values: dict[str, int] = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value, _unit = line.split()
                values[name.rstrip(":")] = int(value) * 1024
        return {
            "rss_bytes": values.get("VmRSS"),
            "peak_rss_bytes": values.get("VmHWM"),
        }
    return {"rss_bytes": None, "peak_rss_bytes": None}


def _sync_cuda(torch: Any) -> None:
    torch.cuda.synchronize()


def _reset_cuda_peak(torch: Any) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated())


def _cuda_memory_payload(torch: Any, baseline_allocated: int) -> dict[str, int]:
    peak_allocated = int(torch.cuda.max_memory_allocated())
    return {
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_incremental_bytes": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _run_model_call(
    *,
    torch: Any,
    model: Any,
    runtime: Any,
    capture_cuda_overlap: bool,
    **model_kwargs: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """Run one model call, optionally preserving a same-origin CUDA timeline."""

    if not capture_cuda_overlap:
        with torch.inference_mode():
            output = model(**model_kwargs)
        _sync_cuda(torch)
        return output, None

    with runtime.cuda_timeline_capture() as capture:
        with torch.inference_mode():
            output = model(**model_kwargs)
        _sync_cuda(torch)
    if not isinstance(capture.result, dict):
        raise RuntimeError("CUDA timeline capture did not produce a result")
    _validate_cuda_timeline_call(capture.result, "CUDA timeline capture")
    return output, capture.result


def _telemetry_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be an object")
    return value


def _telemetry_diagnostic_value(value: object, *, limit: int) -> str:
    """Render untrusted capture diagnostics without dumping arbitrary payloads."""

    if value is None:
        return "null"
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    normalized = " ".join(value.split())
    if len(normalized) > limit:
        normalized = f"{normalized[: limit - 1]}…"
    return repr(normalized)


def _incomplete_timeline_detail(timeline: dict[str, Any]) -> str:
    """Return the runtime's bounded incomplete-capture explanation."""

    return (
        " ("
        f"status={_telemetry_diagnostic_value(timeline.get('status'), limit=64)}; "
        f"reason={_telemetry_diagnostic_value(timeline.get('reason'), limit=240)}"
        ")"
    )


def _telemetry_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return value


def _telemetry_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{name} must be finite and >= 0")
    return result


def _timeline_summary_equal(actual: object, expected: object) -> bool:
    """Compare derived JSON values without treating booleans as integers."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return False
        return all(
            _timeline_summary_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _timeline_summary_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, int)
            and actual == expected
        )
    return type(actual) is type(expected) and actual == expected


def _timeline_span(
    raw: object,
    name: str,
) -> tuple[str, int, CudaInterval]:
    span = _telemetry_mapping(raw, name)
    lane = span.get("lane")
    if lane not in {"h2d", "expert_compute"}:
        raise RuntimeError(f"{name}.lane must be h2d or expert_compute")
    sequence = _telemetry_integer(span.get("sequence"), f"{name}.sequence")
    layer = _telemetry_integer(span.get("layer"), f"{name}.layer")
    expert = _telemetry_integer(span.get("expert"), f"{name}.expert")
    expected_name = f"{lane}:{sequence}:L{layer}:E{expert}"
    if span.get("name") != expected_name:
        raise RuntimeError(f"{name}.name must equal {expected_name!r}")
    start_ms = _telemetry_number(span.get("start_ms"), f"{name}.start_ms")
    end_ms = _telemetry_number(span.get("end_ms"), f"{name}.end_ms")
    if end_ms < start_ms:
        raise RuntimeError(f"{name}.end_ms must be greater than or equal to start_ms")
    duration_ms = _telemetry_number(span.get("duration_ms"), f"{name}.duration_ms")
    if not math.isclose(duration_ms, end_ms - start_ms, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{name}.duration_ms is inconsistent with its endpoints")
    return (
        lane,
        sequence,
        CudaInterval(
            name=expected_name,
            start_ms=start_ms,
            end_ms=end_ms,
        ),
    )


def _validate_timeline_coverage(
    timeline: dict[str, Any],
    *,
    h2d_span_count: int,
    name: str,
) -> None:
    """Require capture-local cache accounting to cover every raw H2D span."""

    coverage = _telemetry_mapping(timeline.get("coverage"), f"{name}.coverage")
    cache_transfer_loads_delta = _telemetry_integer(
        coverage.get("cache_transfer_loads_delta"),
        f"{name}.coverage.cache_transfer_loads_delta",
    )
    reported_h2d_span_count = _telemetry_integer(
        coverage.get("h2d_span_count"),
        f"{name}.coverage.h2d_span_count",
    )
    if reported_h2d_span_count != h2d_span_count:
        raise RuntimeError(
            f"{name}.coverage.h2d_span_count is inconsistent with raw H2D spans"
        )
    if cache_transfer_loads_delta != h2d_span_count:
        raise RuntimeError(
            f"{name}.coverage.cache_transfer_loads_delta is inconsistent with raw H2D spans"
        )


def _require_covered_timeline_schema_version(value: object, name: str) -> None:
    """Reject the legacy timeline contract instead of treating it as coverage."""

    schema_version = _telemetry_integer(value, name)
    if schema_version == _LEGACY_CUDA_TIMELINE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{name}=1 is legacy-unverified; covered CUDA telemetry requires "
            f"schema_version={_CUDA_TIMELINE_SCHEMA_VERSION}"
        )
    if schema_version != _CUDA_TIMELINE_SCHEMA_VERSION:
        raise RuntimeError(f"{name} must be {_CUDA_TIMELINE_SCHEMA_VERSION}")


def _validate_cuda_timeline_call(
    raw: object,
    name: str,
) -> tuple[str, dict[str, Any], str | None]:
    """Fail closed unless one runtime capture matches the covered v2 schema.

    A future worker-aware runtime can append H2D spans on its own stream: the
    schema deliberately validates their metadata and shared-origin timing but
    makes no assumption about their record order or sequence contiguity.
    """

    timeline = _telemetry_mapping(raw, name)
    _require_covered_timeline_schema_version(
        timeline.get("schema_version"),
        f"{name}.schema_version",
    )
    if timeline.get("complete") is not True:
        raise RuntimeError(
            f"{name} must be complete{_incomplete_timeline_detail(timeline)}"
        )
    if timeline.get("method") != _CUDA_TIMELINE_METHOD:
        raise RuntimeError(f"{name}.method is invalid")
    if timeline.get("scope") != _CUDA_TIMELINE_SCOPE:
        raise RuntimeError(f"{name}.scope is invalid")
    if timeline.get("unit") != _CUDA_TIMELINE_UNIT:
        raise RuntimeError(f"{name}.unit is invalid")
    status = timeline.get("status")
    if status not in {"measured", "not_applicable"}:
        raise RuntimeError(f"{name}.status must be measured or not_applicable")
    raw_spans = timeline.get("spans")
    if not isinstance(raw_spans, list):
        raise RuntimeError(f"{name}.spans must be an array")

    transfers: list[CudaInterval] = []
    compute: list[CudaInterval] = []
    sequences: set[int] = set()
    for index, raw_span in enumerate(raw_spans):
        lane, sequence, interval = _timeline_span(raw_span, f"{name}.spans[{index}]")
        if sequence in sequences:
            raise RuntimeError(f"{name}.spans[{index}].sequence must be unique")
        sequences.add(sequence)
        if lane == "h2d":
            transfers.append(interval)
        else:
            compute.append(interval)

    expected_summary = summarize_cuda_timeline(
        transfers=transfers,
        compute=compute,
    )
    expected_summary.pop("intervals")
    if not _timeline_summary_equal(timeline.get("summary"), expected_summary):
        raise RuntimeError(f"{name}.summary is not derived from its raw spans")
    _validate_timeline_coverage(
        timeline,
        h2d_span_count=len(transfers),
        name=name,
    )

    has_both_lanes = bool(transfers) and bool(compute)
    reason = timeline.get("reason")
    if status == "measured":
        if not has_both_lanes:
            raise RuntimeError(f"{name}.status=measured requires both event lanes")
        if reason is not None:
            raise RuntimeError(f"{name}.reason must be null when status=measured")
        return status, expected_summary, None
    if has_both_lanes:
        raise RuntimeError(f"{name}.status=not_applicable has both event lanes")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError(f"{name}.reason is required when status=not_applicable")
    return status, expected_summary, reason


def _aggregate_cuda_overlap(
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate independent model-call captures without mixing their origins."""

    if not calls:
        raise RuntimeError("CUDA overlap telemetry requires at least one model call")
    h2d_interval_count = 0
    compute_interval_count = 0
    h2d_active_durations: list[float] = []
    compute_active_durations: list[float] = []
    overlap_durations: list[float] = []
    active_saved_durations: list[float] = []
    measured_calls = 0
    reasons: list[str] = []
    for index, call in enumerate(calls):
        status, summary, reason = _validate_cuda_timeline_call(
            call,
            f"CUDA timeline capture[{index}]",
        )
        transfer = summary["transfer"]
        compute = summary["compute"]
        overlap = summary["overlap"]
        h2d_interval_count += int(transfer["interval_count"])
        compute_interval_count += int(compute["interval_count"])
        h2d_active_durations.append(float(transfer["active_duration_ms"]))
        compute_active_durations.append(float(compute["active_duration_ms"]))
        overlap_durations.append(float(overlap["duration_ms"]))
        active_saved_durations.append(
            float(overlap["active_duration_saved_by_overlap_ms"])
        )
        if status == "measured":
            measured_calls += 1
        elif reason not in reasons:
            reasons.append(reason)

    status = "measured" if measured_calls else "not_applicable"
    h2d_union_ms = math.fsum(h2d_active_durations)
    compute_union_ms = math.fsum(compute_active_durations)
    overlap_ms = math.fsum(overlap_durations)
    active_saved_ms = math.fsum(active_saved_durations)
    return {
        "schema_version": _CUDA_TIMELINE_SCHEMA_VERSION,
        "status": status,
        "method": _CUDA_TIMELINE_METHOD,
        "scope": _CUDA_TIMELINE_SCOPE,
        "unit": _CUDA_TIMELINE_UNIT,
        "model_call_count": len(calls),
        "measured_model_call_count": measured_calls,
        "h2d_interval_count": h2d_interval_count,
        "expert_compute_interval_count": compute_interval_count,
        "h2d_union_ms": h2d_union_ms,
        "expert_compute_union_ms": compute_union_ms,
        "overlap_ms": overlap_ms,
        "h2d_overlap_fraction": _ratio(overlap_ms, h2d_union_ms),
        "expert_compute_overlap_fraction": _ratio(overlap_ms, compute_union_ms),
        "h2d_hidden_by_compute_ms": overlap_ms,
        "h2d_exposed_ms": max(0.0, h2d_union_ms - overlap_ms),
        "active_duration_saved_by_overlap_ms": active_saved_ms,
        "reason": None if measured_calls else "; ".join(reasons),
        "aggregation": _CUDA_TIMELINE_AGGREGATION,
    }


def _run_inference_pass(
    *,
    label: str,
    torch: Any,
    model: Any,
    tokenizer: Any,
    runtime: Any,
    input_ids: Any,
    attention_mask: Any,
    max_new_tokens: int,
    expert_bytes: int,
    forced_token_ids: list[int] | None = None,
    pipeline_mode: str = "sync",
    capture_cuda_overlap: bool = False,
) -> dict[str, Any]:
    if forced_token_ids is not None and len(forced_token_ids) != max_new_tokens:
        raise ValueError("forced_token_ids must contain exactly max_new_tokens entries")
    gc.collect()
    baseline_allocated = _reset_cuda_peak(torch)
    _sync_cuda(torch)
    rss_before = _process_memory()
    pass_metrics_before = runtime.metrics()
    pass_started = time.perf_counter()

    prefill_metrics_before = runtime.metrics()
    prefill_started = time.perf_counter()
    output, prefill_timeline = _run_model_call(
        torch=torch,
        model=model,
        runtime=runtime,
        capture_cuda_overlap=capture_cuda_overlap,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_metrics = _metrics_delta(runtime.metrics(), prefill_metrics_before)
    _validate_metric_delta(prefill_metrics, expert_bytes=expert_bytes)
    predicted_token = output.logits[:, -1:].argmax(dim=-1)
    past_key_values = output.past_key_values
    del output
    generated_ids = [int(predicted_token.item())]
    fed_token_ids = [
        forced_token_ids[0] if forced_token_ids is not None else generated_ids[0]
    ]
    next_token = torch.tensor(
        [[fed_token_ids[0]]],
        dtype=predicted_token.dtype,
        device=predicted_token.device,
    )
    token_records: list[dict[str, Any]] = [
        {
            "index": 0,
            "token_id": generated_ids[0],
            "predicted_token_id": generated_ids[0],
            "fed_token_id": fed_token_ids[0],
            "matched_forced_token": (
                None
                if forced_token_ids is None
                else generated_ids[0] == fed_token_ids[0]
            ),
            "source": "prefill_to_first_token",
            "latency_seconds": prefill_seconds,
            "metrics": prefill_metrics,
            **(
                {"cuda_event_timeline": prefill_timeline}
                if prefill_timeline is not None
                else {}
            ),
        }
    ]
    cuda_timeline_calls: list[dict[str, Any]] = (
        [prefill_timeline] if prefill_timeline is not None else []
    )
    full_attention_mask = attention_mask
    eos_token_id = tokenizer.eos_token_id

    for token_index in range(1, max_new_tokens):
        if (
            forced_token_ids is None
            and eos_token_id is not None
            and fed_token_ids[-1] == eos_token_id
        ):
            break
        full_attention_mask = torch.cat(
            (
                full_attention_mask,
                torch.ones(
                    (full_attention_mask.shape[0], 1),
                    dtype=full_attention_mask.dtype,
                    device=full_attention_mask.device,
                ),
            ),
            dim=-1,
        )
        token_metrics_before = runtime.metrics()
        _sync_cuda(torch)
        token_started = time.perf_counter()
        output, token_timeline = _run_model_call(
            torch=torch,
            model=model,
            runtime=runtime,
            capture_cuda_overlap=capture_cuda_overlap,
            input_ids=next_token,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        token_seconds = time.perf_counter() - token_started
        predicted_token = output.logits[:, -1:].argmax(dim=-1)
        generated_ids.append(int(predicted_token.item()))
        fed_token_ids.append(
            forced_token_ids[token_index]
            if forced_token_ids is not None
            else generated_ids[-1]
        )
        next_token = torch.tensor(
            [[fed_token_ids[-1]]],
            dtype=predicted_token.dtype,
            device=predicted_token.device,
        )
        past_key_values = output.past_key_values
        del output
        token_records.append(
            {
                "index": token_index,
                "token_id": generated_ids[-1],
                "predicted_token_id": generated_ids[-1],
                "fed_token_id": fed_token_ids[-1],
                "matched_forced_token": (
                    None
                    if forced_token_ids is None
                    else generated_ids[-1] == fed_token_ids[-1]
                ),
                "source": "decode",
                "latency_seconds": token_seconds,
                "metrics": _metrics_delta(runtime.metrics(), token_metrics_before),
                **(
                    {"cuda_event_timeline": token_timeline}
                    if token_timeline is not None
                    else {}
                ),
            }
        )
        if token_timeline is not None:
            cuda_timeline_calls.append(token_timeline)
        _validate_metric_delta(
            token_records[-1]["metrics"],
            expert_bytes=expert_bytes,
        )

    _sync_cuda(torch)
    pass_seconds = time.perf_counter() - pass_started
    pass_metrics = _metrics_delta(runtime.metrics(), pass_metrics_before)
    _validate_metric_delta(pass_metrics, expert_bytes=expert_bytes)
    decode_latencies = [
        float(record["latency_seconds"]) for record in token_records[1:]
    ]
    decode_seconds = sum(decode_latencies)
    matched_forced_tokens = (
        None
        if forced_token_ids is None
        else sum(
            predicted == forced
            for predicted, forced in zip(
                generated_ids,
                fed_token_ids,
                strict=True,
            )
        )
    )
    return {
        "label": label,
        "pipeline": pipeline_mode,
        "cache_state": (
            "cold dynamic expert cache after required static preload"
            if label == "cold_expert_cache"
            else "repeat pass retaining expert cache state from the cold pass"
        ),
        "prefill": {
            "input_tokens": int(input_ids.shape[-1]),
            "wall_seconds": prefill_seconds,
            "metrics": prefill_metrics,
            **(
                {"cuda_event_timeline": prefill_timeline}
                if prefill_timeline is not None
                else {}
            ),
        },
        "decode": {
            "token_count": len(decode_latencies),
            "wall_seconds": decode_seconds,
            "tokens_per_second": _ratio(len(decode_latencies), decode_seconds),
            "latency_p50_seconds": _percentile(decode_latencies, 0.50),
            "latency_p95_seconds": _percentile(decode_latencies, 0.95),
            "per_token": token_records[1:],
        },
        "first_token": token_records[0],
        "total_wall_seconds": pass_seconds,
        "generated_token_count": len(generated_ids),
        "end_to_end_generated_tokens_per_second_including_prefill": (
            len(generated_ids) / pass_seconds
        ),
        "generated_ids": generated_ids,
        "fed_token_ids": fed_token_ids,
        "teacher_forced": forced_token_ids is not None,
        "reference_prediction": (
            None
            if matched_forced_tokens is None
            else {
                "matched_tokens": matched_forced_tokens,
                "total_tokens": len(generated_ids),
                "match_rate": _ratio(matched_forced_tokens, len(generated_ids)),
                "exact_match": matched_forced_tokens == len(generated_ids),
                "first_mismatch_index": next(
                    (
                        index
                        for index, (predicted, forced) in enumerate(
                            zip(generated_ids, fed_token_ids, strict=True)
                        )
                        if predicted != forced
                    ),
                    None,
                ),
            }
        ),
        "generated_text": tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "fed_text": tokenizer.decode(
            fed_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "metrics": pass_metrics,
        "cuda_memory": _cuda_memory_payload(torch, baseline_allocated),
        "process_memory_before": rss_before,
        "process_memory_after": _process_memory(),
        **(
            {"cuda_overlap": _aggregate_cuda_overlap(cuda_timeline_calls)}
            if capture_cuda_overlap
            else {}
        ),
    }


def _package_versions() -> dict[str, str]:
    packages = ("accelerate", "safetensors", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in packages}


def _source_provenance(*, best_effort: bool = False) -> dict[str, str | bool | None]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    script_path = Path(__file__).resolve()
    script_fields = {
        "benchmark_script": script_path.relative_to(_REPO_ROOT).as_posix(),
        "benchmark_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "paged_runtime_sha256": hashlib.sha256(
            (_SRC / "moevm" / "paged_runtime.py").read_bytes()
        ).hexdigest(),
    }
    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        if not best_effort:
            raise RuntimeError(
                "Git provenance is required for benchmark evidence"
            ) from exc
        return {
            "commit": None,
            "tree_clean": None,
            "git_available": False,
            "git_tree_clean_observed": None,
            "provenance_mode": "demo",
            **script_fields,
        }
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Git returned an invalid source commit")
    observed_clean = not bool(status)
    return {
        "commit": commit,
        "tree_clean": None if best_effort else observed_clean,
        "git_available": True,
        "git_tree_clean_observed": observed_clean,
        "provenance_mode": "demo" if best_effort else "benchmark_evidence",
        **script_fields,
    }


def _total_checkpoint_bytes(snapshot: Path) -> int:
    payload = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint index metadata is missing")
    total_size = metadata.get("total_size")
    if (
        isinstance(total_size, bool)
        or not isinstance(total_size, int)
        or total_size <= 0
    ):
        raise ValueError("checkpoint index total_size is invalid")
    return total_size


def _validate_model_shape(config: Any, store: Any) -> None:
    if config.model_type != "olmoe":
        raise ValueError(f"expected olmoe config, got {config.model_type}")
    if tuple(range(config.num_hidden_layers)) != store.layers:
        raise ValueError("checkpoint expert layers do not match the model config")
    if any(
        len(store.experts_in_layer(layer)) != config.num_experts
        for layer in store.layers
    ):
        raise ValueError("checkpoint expert counts do not match the model config")
    if (
        config.hidden_size != store.spec.hidden_size
        or config.intermediate_size != store.spec.intermediate_size
    ):
        raise ValueError("checkpoint expert shapes do not match the model config")
    if config.num_hidden_layers != 16:
        raise ValueError("pinned OLMoE must have exactly 16 layers")
    if config.num_experts != 64 or config.num_experts_per_tok != 8:
        raise ValueError("pinned OLMoE must have 64 experts and top-8 routing")
    if config.hidden_size != 2048 or config.intermediate_size != 1024:
        raise ValueError("pinned OLMoE dimensions are not 2048/1024")
    if config.hidden_act != "silu":
        raise ValueError("pinned OLMoE must use SiLU experts")


def _write_json_create_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    source = _source_provenance(best_effort=args.demo_mode)
    if source["tree_clean"] is not True and not args.demo_mode:
        raise RuntimeError("benchmark evidence requires a clean Git working tree")
    prompt = _resolve_prompt(args)
    pipeline_profile = None
    pipeline_profile_sha256 = None
    if args.pipeline == "auto":
        pipeline_profile, pipeline_profile_sha256 = load_pipeline_profile(
            Path(args.pipeline_profile)
        )
    snapshot = _validate_snapshot(Path(args.snapshot))
    output_path = Path(args.output).expanduser().resolve()
    if output_path.is_relative_to(snapshot):
        raise ValueError("output must not be written inside the read-only snapshot")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    shard_sizes = _validate_pinned_shard_files(snapshot)
    reference_metadata = None
    if args.reference_metadata:
        reference_metadata = _load_reference_metadata(
            Path(args.reference_metadata),
            workload_id=args.workload_id,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
    forced_token_ids = None
    if args.teacher_force_reference:
        forced_token_ids = reference_metadata["generated_token_ids"]
        if len(forced_token_ids) != args.max_new_tokens:
            raise ValueError(
                "teacher-forced reference must contain at least max_new_tokens IDs"
            )

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from moevm.paged_runtime import (
        CachePolicy,
        ExpertSlotCache,
        PagedExpertRuntime,
        SafetensorExpertStore,
        attach_transformers_olmoe_runtime,
        load_non_expert_weights_into_meta_model,
        validate_transformers_paged_model,
    )
    from moevm.types import ExpertKey

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full paged OLMoE harness")
    if importlib.metadata.version("transformers") != "5.14.1":
        raise RuntimeError("the paged integration requires transformers==5.14.1")
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    device_uuid = str(torch.cuda.get_device_properties(device).uuid).lower()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device must support BF16")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = None
    runtime = None
    cache = None
    try:
        total_load_started = time.perf_counter()
        with ExitStack() as stack:
            store = stack.enter_context(SafetensorExpertStore(snapshot))
            config_started = time.perf_counter()
            config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            _validate_model_shape(config, store)
            if store.spec.dtype != torch.bfloat16:
                raise RuntimeError("the pinned OLMoE expert store must be BF16")
            config_seconds = time.perf_counter() - config_started

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(device)
            if input_ids.numel() == 0:
                raise RuntimeError("tokenizer produced no input tokens")

            hotset_digest = None
            hotsets: dict[int, tuple[int, ...]] = {layer: () for layer in store.layers}
            if args.policy == "hybrid":
                hotsets, hotset_digest = _load_hotsets(
                    Path(args.hotset_json),
                    layers=store.layers,
                    experts_per_layer=config.num_experts,
                    slots_per_layer=args.slots_per_layer,
                )
            static_keys = tuple(
                ExpertKey(layer, expert)
                for layer in store.layers
                for expert in hotsets[layer]
            )

            total_checkpoint_bytes = _total_checkpoint_bytes(snapshot)
            total_expert_bytes = len(store) * store.spec.size_bytes
            non_expert_bytes = total_checkpoint_bytes - total_expert_bytes
            cache_bytes = (
                len(store.layers) * args.slots_per_layer * store.spec.size_bytes
            )
            expected_weight_bytes = cache_bytes + non_expert_bytes
            free_before, total_vram = torch.cuda.mem_get_info()
            required_free = expected_weight_bytes + _VRAM_SAFETY_MARGIN_BYTES
            if free_before < required_free:
                raise RuntimeError(
                    "insufficient free VRAM for guarded allocation: "
                    f"{free_before} < {required_free} bytes"
                )
            budget = {
                "slots_per_layer": args.slots_per_layer,
                "layers": len(store.layers),
                "expert_bytes": store.spec.size_bytes,
                "cache_bytes": cache_bytes,
                "staging_slots": args.staging_slots,
                "staging_host_bytes": args.staging_slots * store.spec.size_bytes,
                "pipeline": args.pipeline,
                "non_expert_checkpoint_bytes": non_expert_bytes,
                "expected_weight_vram_bytes": expected_weight_bytes,
                "safety_margin_bytes": _VRAM_SAFETY_MARGIN_BYTES,
                "required_free_vram_bytes": required_free,
                "observed_free_vram_bytes": int(free_before),
                "device_total_vram_bytes": int(total_vram),
            }

            package_versions = _package_versions()
            if pipeline_profile is None:
                pipeline_schedule = {
                    "cold_expert_cache": args.pipeline,
                    "repeat_retained_expert_cache": args.pipeline,
                }
            else:
                expected_binding = result_binding(
                    {
                        "model": {
                            "model_id": PINNED_MODEL_ID,
                            "revision": PINNED_REVISION,
                            "dtype": str(store.spec.dtype),
                            "layers": len(store.layers),
                            "experts_per_layer": config.num_experts,
                            "top_k": config.num_experts_per_tok,
                            "shards": {
                                name: {"sha256": digest}
                                for name, digest in sorted(PINNED_SHARD_SHA256.items())
                            },
                        },
                        "runtime": {
                            "device_uuid": device_uuid,
                            "device_name": device_name,
                            "policy": args.policy,
                            "capacity_scope": "independent per-layer partitions",
                            "hotset_sha256": hotset_digest,
                            "budget": budget,
                        },
                        "workload": {
                            "id": args.workload_id,
                            "prompt_sha256": _prompt_sha256(prompt),
                            "input_ids": [
                                int(token) for token in input_ids[0].tolist()
                            ],
                            "input_tokens": int(input_ids.shape[-1]),
                            "max_new_tokens": args.max_new_tokens,
                            "decoding": (
                                "teacher-forced reference with greedy predictions"
                                if args.teacher_force_reference
                                else "greedy"
                            ),
                            "seed": args.seed,
                        },
                        "source": source,
                        "environment": {
                            "python": platform.python_version(),
                            "platform": platform.platform(),
                            "packages": package_versions,
                        },
                    }
                )
                validate_profile_binding(pipeline_profile, expected_binding)
                pipeline_schedule = dict(pipeline_profile["selection"])
            budget["resolved_pipeline_by_pass"] = pipeline_schedule

            cache_initial_mode = args.pipeline
            if args.pipeline == "auto":
                cache_initial_mode = (
                    "async" if "async" in pipeline_schedule.values() else "sync"
                )

            model_load_baseline = _reset_cuda_peak(torch)
            rss_before_load = _process_memory()
            meta_started = time.perf_counter()
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(config)
            meta_seconds = time.perf_counter() - meta_started

            cache_started = time.perf_counter()
            cache = ExpertSlotCache(
                store,
                capacity_per_layer=args.slots_per_layer,
                device=device,
                policy=(
                    CachePolicy.HYBRID if args.policy == "hybrid" else CachePolicy.LRU
                ),
                static_keys=static_keys,
                staging_slots=args.staging_slots,
                pin_staging=True,
                pipeline_mode=cache_initial_mode,
            )
            stack.callback(cache.close)
            if args.pipeline == "auto":
                cache.set_pipeline_mode(pipeline_schedule["cold_expert_cache"])
            runtime = PagedExpertRuntime(cache)
            attach_transformers_olmoe_runtime(model, runtime)
            cache_allocation_seconds = time.perf_counter() - cache_started

            non_expert_started = time.perf_counter()
            loaded_non_expert = load_non_expert_weights_into_meta_model(
                model,
                store,
                device=device,
            )
            _sync_cuda(torch)
            non_expert_load_seconds = time.perf_counter() - non_expert_started
            validation = validate_transformers_paged_model(
                model,
                store,
                device=device,
                runtime=runtime,
            )
            model.eval()

            preload_before = runtime.metrics()
            preload_started = time.perf_counter()
            if static_keys:
                cache.prefetch(static_keys)
            _sync_cuda(torch)
            preload_seconds = time.perf_counter() - preload_started
            preload_metrics = _metrics_delta(runtime.metrics(), preload_before)
            _validate_metric_delta(
                preload_metrics,
                expert_bytes=store.spec.size_bytes,
            )
            model_load_seconds = time.perf_counter() - total_load_started
            model_load = {
                "total_seconds": model_load_seconds,
                "config_tokenizer_seconds": config_seconds,
                "meta_initialization_seconds": meta_seconds,
                "cache_allocation_seconds": cache_allocation_seconds,
                "non_expert_load_seconds": non_expert_load_seconds,
                "static_preload_seconds": preload_seconds,
                "static_preload_metrics": preload_metrics,
                "loaded_non_expert_tensors": len(loaded_non_expert),
                "validation": validation,
                "cuda_memory": _cuda_memory_payload(torch, model_load_baseline),
                "process_memory_before": rss_before_load,
                "process_memory_after": _process_memory(),
            }

            cold = _run_inference_pass(
                label="cold_expert_cache",
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                runtime=runtime,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                expert_bytes=store.spec.size_bytes,
                forced_token_ids=forced_token_ids,
                pipeline_mode=pipeline_schedule["cold_expert_cache"],
                capture_cuda_overlap=args.cuda_overlap_telemetry,
            )
            gc.collect()
            torch.cuda.empty_cache()
            if args.pipeline == "auto":
                cache.set_pipeline_mode(
                    pipeline_schedule["repeat_retained_expert_cache"]
                )
            warm = _run_inference_pass(
                label="repeat_retained_expert_cache",
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                runtime=runtime,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                expert_bytes=store.spec.size_bytes,
                forced_token_ids=forced_token_ids,
                pipeline_mode=pipeline_schedule["repeat_retained_expert_cache"],
                capture_cuda_overlap=args.cuda_overlap_telemetry,
            )
            if cold["generated_ids"] != warm["generated_ids"]:
                raise RuntimeError("cold and warm greedy outputs differ")
            reference_comparison: dict[str, Any]
            if reference_metadata is None:
                reference_comparison = {
                    "available": False,
                    "matched": None,
                    "reason": "no --reference-metadata supplied",
                }
            else:
                reference_ids = reference_metadata["generated_token_ids"]
                actual_prefix = cold["generated_ids"][: len(reference_ids)]
                if actual_prefix != reference_ids and not args.teacher_force_reference:
                    raise RuntimeError(
                        "paged greedy token IDs differ from the pinned baseline prefix"
                    )
                reference_comparison = {
                    "available": True,
                    "matched": actual_prefix == reference_ids,
                    "mode": (
                        "teacher_forced"
                        if args.teacher_force_reference
                        else "autoregressive_exact_gate"
                    ),
                    "matched_tokens": sum(
                        actual == expected
                        for actual, expected in zip(
                            actual_prefix,
                            reference_ids,
                            strict=True,
                        )
                    ),
                    "total_tokens": len(reference_ids),
                    "first_mismatch_index": next(
                        (
                            index
                            for index, (actual, expected) in enumerate(
                                zip(actual_prefix, reference_ids, strict=True)
                            )
                            if actual != expected
                        ),
                        None,
                    ),
                    **reference_metadata,
                }

            # No expert reads occur after the two inference passes.  The helper
            # releases persistent mappings before the independent integrity gate.
            cache.wait_idle()
            cache.close()
            verification_started = time.perf_counter()
            shard_verification = _close_store_and_verify_pinned_shards(
                store,
                snapshot,
            )
            verification_seconds = time.perf_counter() - verification_started
            comparison = {
                "repeat_over_cold_speedup": _ratio(
                    float(cold["total_wall_seconds"]),
                    float(warm["total_wall_seconds"]),
                ),
                "repeat_wall_reduction": (
                    None
                    if float(cold["total_wall_seconds"]) == 0.0
                    else 1.0
                    - float(warm["total_wall_seconds"])
                    / float(cold["total_wall_seconds"])
                ),
                "repeat_throughput_change": (
                    _ratio(
                        float(
                            warm[
                                "end_to_end_generated_tokens_per_second_including_prefill"
                            ]
                        ),
                        float(
                            cold[
                                "end_to_end_generated_tokens_per_second_including_prefill"
                            ]
                        ),
                    )
                ),
                "hit_rate_point_change": (
                    float(warm["metrics"]["hit_rate"])
                    - float(cold["metrics"]["hit_rate"])
                ),
                "logical_storage_traffic_ratio": _ratio(
                    float(warm["metrics"]["storage_bytes"]),
                    float(cold["metrics"]["storage_bytes"]),
                ),
            }
            if comparison["repeat_throughput_change"] is not None:
                comparison["repeat_throughput_change"] -= 1.0

            return {
                "schema_version": 1,
                "status": "ok",
                "created_at": datetime.now(UTC).isoformat(),
                "evidence": {
                    "label": (
                        "interactive local hardware demo; not benchmark evidence"
                        if args.demo_mode
                        else "single-workload controlled paged-runtime smoke; "
                        "not a general throughput claim"
                    ),
                    "publishable_benchmark_evidence": not args.demo_mode,
                    "offline_local_only": True,
                    "limitations": [
                        *(
                            [
                                "Demo mode permits best-effort Git provenance and must not be published as benchmark evidence."
                            ]
                            if args.demo_mode
                            else []
                        ),
                        "SHA-256 verification is intentionally after timed passes to avoid warming every shard first.",
                        "Model loading and mmap page faults still make OS page-cache state uncontrolled.",
                        "Cold means an empty dynamic expert cache, not a cold NVMe or OS cache.",
                        "Storage time includes safetensors mmap/page faults and CPU staging copies.",
                        *(
                            [
                                "CUDA overlap telemetry uses timing events on one GPU to measure paged-expert H2D and paged-expert compute intervals. It does not establish physical NVMe activity, page-cache state, or a general speedup.",
                                "CUDA overlap telemetry is instrumentation; compare wall time only against paired runs collected with the same telemetry setting.",
                            ]
                            if args.cuda_overlap_telemetry
                            else []
                        ),
                        (
                            "The Python runtime is synchronous and does not overlap storage, H2D, or compute."
                            if args.pipeline == "sync"
                            else (
                                "The bounded async path is designed to permit mmap/page-cache service and pinned H2D to progress alongside expert compute inside a routed layer; this run does not itself measure interval overlap or prove physical NVMe activity."
                                if args.pipeline == "async"
                                else (
                                    "The conservative adaptive path selects async only for routed-layer calls that start with a free eligible slot and at least two requested misses; calls starting full use sync. This rule is experimental and not a performance guarantee."
                                    if args.pipeline == "adaptive"
                                    else "Auto uses a measured profile bound to this GPU, model, workload, cache budget, and exact benchmark/runtime code. Calibration remains workload-specific and is not a universal performance guarantee."
                                )
                            )
                        ),
                        "One prompt and at most a few greedy tokens cannot establish production performance.",
                        "Prefill groups unique active experts per layer; its lookups are not a tokenwise trace replay.",
                        "No model.to/cuda/cpu, torch.compile, or graph capture is allowed after partial meta loading.",
                        (
                            "Teacher-forced mode measures a fixed reference prefix and does not establish autoregressive output identity."
                            if args.teacher_force_reference
                            else "The default reference gate requires exact autoregressive token identity."
                        ),
                    ],
                },
                "model": {
                    "model_id": PINNED_MODEL_ID,
                    "revision": PINNED_REVISION,
                    "snapshot": str(snapshot),
                    "shards": shard_verification,
                    "preflight_shard_sizes": shard_sizes,
                    "hash_verification_seconds": verification_seconds,
                    "dtype": str(store.spec.dtype),
                    "layers": len(store.layers),
                    "experts_per_layer": config.num_experts,
                    "top_k": config.num_experts_per_tok,
                },
                "runtime": {
                    "device": str(device),
                    "device_name": device_name,
                    "device_uuid": device_uuid,
                    "policy": args.policy,
                    "pipeline": args.pipeline,
                    "resolved_pipeline_by_pass": pipeline_schedule,
                    "pipeline_profile_sha256": pipeline_profile_sha256,
                    "pipeline_profile_calibration_pairs": (
                        None
                        if pipeline_profile is None
                        else pipeline_profile["calibration"]["pairs"]
                    ),
                    "cuda_overlap_telemetry": {
                        "requested": args.cuda_overlap_telemetry,
                        "method": (
                            "cuda_events_v1" if args.cuda_overlap_telemetry else None
                        ),
                        "scope": (
                            "paged_expert_h2d_vs_expert_compute"
                            if args.cuda_overlap_telemetry
                            else None
                        ),
                    },
                    "capacity_scope": "independent per-layer partitions",
                    "hotset_json": args.hotset_json,
                    "hotset_sha256": hotset_digest,
                    "protected_hot_per_layer": {
                        str(layer): len(hotsets[layer]) for layer in store.layers
                    },
                    "budget": budget,
                    "final_metrics": _metrics_payload(runtime.metrics()),
                },
                "workload": {
                    "id": args.workload_id,
                    "prompt": prompt,
                    "prompt_sha256": _prompt_sha256(prompt),
                    "input_ids": [int(token) for token in input_ids[0].tolist()],
                    "input_tokens": int(input_ids.shape[-1]),
                    "max_new_tokens": args.max_new_tokens,
                    "decoding": (
                        "teacher-forced reference with greedy predictions"
                        if args.teacher_force_reference
                        else "greedy"
                    ),
                    "seed": args.seed,
                },
                "model_load": model_load,
                "passes": {
                    "cold_expert_cache": cold,
                    "repeat_retained_expert_cache": warm,
                },
                "repeat_comparison": comparison,
                "reference_comparison": reference_comparison,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": package_versions,
                },
                "source": source,
            }
    finally:
        if "torch" in locals() and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
        if model is not None:
            try:
                for layer in model.model.layers:
                    layer.mlp.experts._moevm_paged_runtime = None
            except AttributeError:
                pass
        model = None
        runtime = None
        cache = None
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_cuda_oom(error: BaseException) -> bool:
    return error.__class__.__name__ == "OutOfMemoryError" or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        report = run_benchmark(args)
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        report = {
            "schema_version": 1,
            "status": "cuda_oom",
            "created_at": datetime.now(UTC).isoformat(),
            "model": {
                "model_id": PINNED_MODEL_ID,
                "revision": PINNED_REVISION,
            },
            "error": str(exc),
            "evidence": {
                "label": "failed guarded paged-runtime smoke; no performance result"
            },
        }
        _write_json_create_only(Path(args.output), report)
        print(f"CUDA OOM; wrote failure evidence to {args.output}", file=sys.stderr)
        return 2
    _write_json_create_only(Path(args.output), report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
