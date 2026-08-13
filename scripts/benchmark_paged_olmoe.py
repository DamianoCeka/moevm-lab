#!/usr/bin/env python3
"""Run one controlled offline OLMoE paged-runtime cold/warm smoke benchmark."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PINNED_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PINNED_REVISION = "bd1c52f59153f724c1ad11ca1791edc77bab3806"
PINNED_SHARD_SHA256 = {
    "model-00001-of-00003.safetensors": (
        "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"
    ),
    "model-00002-of-00003.safetensors": (
        "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"
    ),
    "model-00003-of-00003.safetensors": (
        "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"
    ),
}
PINNED_SHARD_SIZES = {
    "model-00001-of-00003.safetensors": 4_997_744_872,
    "model-00002-of-00003.safetensors": 4_997_235_176,
    "model-00003-of-00003.safetensors": 3_843_741_912,
}
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
_TIME_METRICS = (
    "storage_seconds",
    "transfer_seconds",
    "forward_seconds",
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
        type=_bounded_integer("max-new-tokens", 1, 16),
        default=2,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--reference-metadata",
        help="Optional v1 pinned greedy-baseline metadata used as a correctness gate.",
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
    expected_bytes = int(payload["misses"]) * expert_bytes
    if int(payload["storage_bytes"]) != expected_bytes:
        raise RuntimeError("cache metric invariant failed for logical storage bytes")
    if int(payload["host_to_device_bytes"]) != expected_bytes:
        raise RuntimeError("cache metric invariant failed for logical H2D bytes")
    if any(float(payload[name]) < 0.0 for name in (*_INTEGER_METRICS, *_TIME_METRICS)):
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
) -> dict[str, Any]:
    gc.collect()
    baseline_allocated = _reset_cuda_peak(torch)
    _sync_cuda(torch)
    rss_before = _process_memory()
    pass_metrics_before = runtime.metrics()
    pass_started = time.perf_counter()

    prefill_metrics_before = runtime.metrics()
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    _sync_cuda(torch)
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_metrics = _metrics_delta(runtime.metrics(), prefill_metrics_before)
    _validate_metric_delta(prefill_metrics, expert_bytes=expert_bytes)
    next_token = output.logits[:, -1:].argmax(dim=-1)
    past_key_values = output.past_key_values
    del output
    generated_ids = [int(next_token.item())]
    token_records: list[dict[str, Any]] = [
        {
            "index": 0,
            "token_id": generated_ids[0],
            "source": "prefill_to_first_token",
            "latency_seconds": prefill_seconds,
            "metrics": prefill_metrics,
        }
    ]
    full_attention_mask = attention_mask
    eos_token_id = tokenizer.eos_token_id

    for token_index in range(1, max_new_tokens):
        if eos_token_id is not None and generated_ids[-1] == eos_token_id:
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
        with torch.inference_mode():
            output = model(
                input_ids=next_token,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        _sync_cuda(torch)
        token_seconds = time.perf_counter() - token_started
        next_token = output.logits[:, -1:].argmax(dim=-1)
        generated_ids.append(int(next_token.item()))
        past_key_values = output.past_key_values
        del output
        token_records.append(
            {
                "index": token_index,
                "token_id": generated_ids[-1],
                "source": "decode",
                "latency_seconds": token_seconds,
                "metrics": _metrics_delta(runtime.metrics(), token_metrics_before),
            }
        )
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
    return {
        "label": label,
        "cache_state": (
            "cold dynamic expert cache after required static preload"
            if label == "cold_expert_cache"
            else "repeat pass retaining expert cache state from the cold pass"
        ),
        "prefill": {
            "input_tokens": int(input_ids.shape[-1]),
            "wall_seconds": prefill_seconds,
            "metrics": prefill_metrics,
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
        "generated_text": tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "metrics": pass_metrics,
        "cuda_memory": _cuda_memory_payload(torch, baseline_allocated),
        "process_memory_before": rss_before,
        "process_memory_after": _process_memory(),
    }


def _package_versions() -> dict[str, str]:
    packages = ("accelerate", "safetensors", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in packages}


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
    prompt = _resolve_prompt(args)
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
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device must support BF16")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = None
    runtime = None
    cache = None
    try:
        total_load_started = time.perf_counter()
        with SafetensorExpertStore(snapshot) as store:
            config_started = time.perf_counter()
            config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            _validate_model_shape(config, store)
            if store.spec.dtype != torch.bfloat16:
                raise RuntimeError("the pinned OLMoE expert store must be BF16")
            config_seconds = time.perf_counter() - config_started

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
                "non_expert_checkpoint_bytes": non_expert_bytes,
                "expected_weight_vram_bytes": expected_weight_bytes,
                "safety_margin_bytes": _VRAM_SAFETY_MARGIN_BYTES,
                "required_free_vram_bytes": required_free,
                "observed_free_vram_bytes": int(free_before),
                "device_total_vram_bytes": int(total_vram),
            }

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
            )
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
            )
            gc.collect()
            torch.cuda.empty_cache()
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
                if actual_prefix != reference_ids:
                    raise RuntimeError(
                        "paged greedy token IDs differ from the pinned baseline prefix"
                    )
                reference_comparison = {
                    "available": True,
                    "matched": True,
                    **reference_metadata,
                }

            # No expert reads occur after the two inference passes.  The helper
            # releases persistent mappings before the independent integrity gate.
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
                        "single-workload controlled paged-runtime smoke; "
                        "not a general throughput claim"
                    ),
                    "offline_local_only": True,
                    "limitations": [
                        "SHA-256 verification is intentionally after timed passes to avoid warming every shard first.",
                        "Model loading and mmap page faults still make OS page-cache state uncontrolled.",
                        "Cold means an empty dynamic expert cache, not a cold NVMe or OS cache.",
                        "Storage time includes safetensors mmap/page faults and CPU staging copies.",
                        "The Python runtime is synchronous and does not overlap storage, H2D, or compute.",
                        "One prompt and at most a few greedy tokens cannot establish production performance.",
                        "Prefill groups unique active experts per layer; its lookups are not a tokenwise trace replay.",
                        "No model.to/cuda/cpu, torch.compile, or graph capture is allowed after partial meta loading.",
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
                    "device_name": torch.cuda.get_device_name(),
                    "policy": args.policy,
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
                    "decoding": "greedy",
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
                    "packages": _package_versions(),
                },
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
