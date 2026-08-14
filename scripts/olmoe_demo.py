#!/usr/bin/env python3
"""Run a resumable, one-command OLMoE paging demo.

The module intentionally imports only the Python standard library at startup.
Checkpoint discovery and verification are delegated to ``moevm.olmoe_assets``
from :func:`main`, after command-line parsing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_paged_olmoe.py"
PAGED_RUNTIME_SOURCE = SRC_ROOT / "moevm" / "paged_runtime.py"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "huggingface"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "demo"
DEFAULT_WORKLOAD_FILE = REPO_ROOT / "benchmarks" / "workloads" / "olmoe_m1.json"

LAYER_COUNT = 16
EXPERTS_PER_LAYER = 64
EXPERT_BYTES = 12 * 1024**2
SLOT_FOOTPRINT_BYTES = LAYER_COUNT * EXPERT_BYTES
MIN_SLOTS_PER_LAYER = 2
MAX_SLOTS_PER_LAYER = 32
MIN_VRAM_RESERVE_BYTES = 2 * 1024**3
VRAM_RESERVE_FRACTION = 0.20
DEFAULT_SEED = 17

PASS_NAMES = ("cold_expert_cache", "repeat_retained_expert_cache")
COUNTER_FIELDS = (
    "requests",
    "hits",
    "misses",
    "evictions",
    "storage_loads",
    "transfer_loads",
    "storage_bytes",
    "host_to_device_bytes",
    "coalesced_requests",
    "admission_rejections",
    "storage_failures",
    "transfer_failures",
)
ZERO_SAFETY_COUNTERS = (
    "admission_rejections",
    "storage_failures",
    "transfer_failures",
)
SUMMARY_LIMITATIONS = (
    "This is a short local smoke measurement, not a production throughput claim.",
    "The operating-system page cache and background machine load are uncontrolled.",
    "Logical storage bytes do not prove physical NVMe activity.",
    "Wall time alone does not prove physical NVMe or CUDA interval overlap.",
    "Process working set can include reclaimable mmap-backed checkpoint pages.",
)


class DemoError(RuntimeError):
    """A user-facing, fail-closed demo error."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class ProcessRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult: ...


class SubprocessRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=None if env is None else dict(env),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        return ProcessResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    total_bytes: int
    free_bytes: int
    compute_capability: str
    driver_version: str


@dataclass(frozen=True)
class DemoConfig:
    snapshot: Path
    output_root: Path
    workload_file: Path
    workload_id: str
    device: str
    requested_slots: str
    max_input_tokens: int = 64
    max_new_tokens: int = 2
    compare: bool = False
    dry_run: bool = False
    python_executable: Path = Path(sys.executable)
    benchmark_script: Path = BENCHMARK_SCRIPT


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DemoError(f"{name} must be a JSON object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise DemoError(f"{name} must be a JSON array")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DemoError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        relation = "> 0" if positive else ">= 0"
        raise DemoError(f"{name} must be finite and {relation}")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise DemoError(f"{name} must be a string")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_json_create_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.write("\n")


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoError(f"cannot read valid {name} JSON from {path}: {exc}") from exc
    return _mapping(payload, name)


def _device_index(device: str) -> int:
    if device == "cuda":
        return 0
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match is None:
        raise DemoError("device must be 'cuda' or 'cuda:<index>'")
    return int(match.group(1))


def probe_gpu(runner: ProcessRunner, device: str) -> GpuInfo:
    requested_index = _device_index(device)
    argv = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = runner.run(argv, cwd=REPO_ROOT)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DemoError(f"nvidia-smi failed: {detail or 'no diagnostic output'}")

    matches: list[GpuInfo] = []
    try:
        rows = csv.reader(result.stdout.splitlines(), skipinitialspace=True)
        for row in rows:
            if len(row) != 7:
                continue
            index = int(row[0].strip())
            total_mib = int(row[3].strip())
            free_mib = int(row[4].strip())
            if index == requested_index:
                uuid = row[1].strip().lower().removeprefix("gpu-")
                matches.append(
                    GpuInfo(
                        index=index,
                        uuid=uuid,
                        name=row[2].strip(),
                        total_bytes=total_mib * 1024**2,
                        free_bytes=free_mib * 1024**2,
                        compute_capability=row[5].strip(),
                        driver_version=row[6].strip(),
                    )
                )
    except ValueError as exc:
        raise DemoError("nvidia-smi returned malformed numeric fields") from exc
    if len(matches) != 1:
        raise DemoError(f"CUDA device index {requested_index} was not found uniquely")
    gpu = matches[0]
    if gpu.total_bytes <= 0 or not 0 <= gpu.free_bytes <= gpu.total_bytes:
        raise DemoError("nvidia-smi returned an invalid VRAM capacity")
    return gpu


def checkpoint_sizes(snapshot: Path) -> tuple[int, int]:
    index_path = snapshot / "model.safetensors.index.json"
    payload = _load_json(index_path, "checkpoint index")
    metadata = _mapping(payload.get("metadata"), "checkpoint index metadata")
    total_bytes = _integer(
        metadata.get("total_size"),
        "checkpoint index metadata.total_size",
        minimum=1,
    )
    total_expert_bytes = LAYER_COUNT * EXPERTS_PER_LAYER * EXPERT_BYTES
    non_expert_bytes = total_bytes - total_expert_bytes
    if non_expert_bytes <= 0:
        raise DemoError("checkpoint index is too small for the pinned expert topology")
    return total_bytes, non_expert_bytes


def plan_slots(
    gpu: GpuInfo,
    non_expert_bytes: int,
    requested: str,
) -> tuple[int, int, int]:
    reserve_bytes = max(
        MIN_VRAM_RESERVE_BYTES,
        int(gpu.total_bytes * VRAM_RESERVE_FRACTION),
    )
    usable_bytes = gpu.free_bytes - reserve_bytes - non_expert_bytes
    affordable = usable_bytes // SLOT_FOOTPRINT_BYTES
    if requested == "auto":
        if affordable < MIN_SLOTS_PER_LAYER:
            required = (
                reserve_bytes
                + non_expert_bytes
                + MIN_SLOTS_PER_LAYER * SLOT_FOOTPRINT_BYTES
            )
            raise DemoError(
                "insufficient free VRAM for automatic two-slot planning: "
                f"{gpu.free_bytes} available, {required} required"
            )
        automatic = min(
            MAX_SLOTS_PER_LAYER,
            max(MIN_SLOTS_PER_LAYER, affordable),
        )
        selected = automatic
    else:
        try:
            selected = int(requested)
        except ValueError as exc:
            raise DemoError("slots must be 'auto' or an integer from 1 to 32") from exc
        if not 1 <= selected <= MAX_SLOTS_PER_LAYER:
            raise DemoError("slots must be 'auto' or an integer from 1 to 32")
        if selected > affordable:
            raise DemoError(
                f"requested {selected} slots per layer, but only {affordable} fit "
                "the guarded free-VRAM budget"
            )
    return selected, reserve_bytes, affordable


def _validate_workload(path: Path, workload_id: str) -> str:
    payload = _load_json(path, "workload collection")
    if payload.get("schema_version") != 1:
        raise DemoError("workload collection schema_version must be 1")
    workloads = _array(payload.get("workloads"), "workload collection.workloads")
    prompts = [
        item.get("prompt")
        for item in workloads
        if isinstance(item, dict) and item.get("id") == workload_id
    ]
    if len(prompts) != 1 or not isinstance(prompts[0], str) or not prompts[0].strip():
        raise DemoError(f"workload id must match exactly one prompt: {workload_id}")
    return prompts[0]


def build_plan(
    config: DemoConfig,
    *,
    model_id: str,
    revision: str,
    shard_sha256: Mapping[str, str],
    gpu: GpuInfo,
    slots_per_layer: int,
    reserve_bytes: int,
    checkpoint_bytes: int,
    non_expert_bytes: int,
) -> dict[str, Any]:
    prompt = _validate_workload(config.workload_file, config.workload_id)
    core: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "id": model_id,
            "revision": revision,
            "checkpoint_bytes": checkpoint_bytes,
            "non_expert_bytes": non_expert_bytes,
            "shards_sha256": dict(sorted(shard_sha256.items())),
        },
        "workload": {
            "id": config.workload_id,
            "collection_sha256": _sha256(config.workload_file),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        "runtime": {
            "device": config.device,
            "gpu_index": gpu.index,
            "gpu_uuid_sha256": hashlib.sha256(gpu.uuid.encode("utf-8")).hexdigest(),
            "gpu_name": gpu.name,
            "gpu_total_vram_bytes": gpu.total_bytes,
            "gpu_compute_capability": gpu.compute_capability,
            "nvidia_driver_version": gpu.driver_version,
            "pipeline_modes": ["sync", "async"] if config.compare else ["async"],
            "slots_per_layer": slots_per_layer,
            "slot_footprint_bytes": SLOT_FOOTPRINT_BYTES,
            "cache_bytes": slots_per_layer * SLOT_FOOTPRINT_BYTES,
            "staging_slots": 2,
            "vram_reserve_bytes": reserve_bytes,
        },
        "generation": {
            "max_input_tokens": config.max_input_tokens,
            "max_new_tokens": config.max_new_tokens,
            "seed": DEFAULT_SEED,
        },
        "source": {
            "benchmark_script_sha256": _sha256(config.benchmark_script),
            "paged_runtime_sha256": _sha256(PAGED_RUNTIME_SOURCE),
        },
    }
    plan_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()[:20]
    return {"id": plan_id, **core}


def _validate_plan(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    if payload != expected:
        raise DemoError("existing plan.json does not match the requested demo plan")
    core = {key: value for key, value in payload.items() if key != "id"}
    expected_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()[:20]
    if payload.get("id") != expected_id:
        raise DemoError("existing plan.json has an invalid deterministic id")


def benchmark_argv(
    config: DemoConfig,
    *,
    mode: str,
    slots_per_layer: int,
    output: Path,
) -> list[str]:
    if mode not in {"sync", "async"}:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    return [
        str(config.python_executable),
        str(config.benchmark_script),
        "--demo-mode",
        "--snapshot",
        str(config.snapshot),
        "--output",
        str(output),
        "--workload-file",
        str(config.workload_file),
        "--workload-id",
        config.workload_id,
        "--device",
        config.device,
        "--policy",
        "lru",
        "--pipeline",
        mode,
        "--slots-per-layer",
        str(slots_per_layer),
        "--staging-slots",
        "2",
        "--max-input-tokens",
        str(config.max_input_tokens),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--seed",
        str(DEFAULT_SEED),
    ]


def _validate_metrics(value: object, name: str) -> dict[str, int]:
    metrics = _mapping(value, name)
    result = {
        field: _integer(metrics.get(field), f"{name}.{field}")
        for field in COUNTER_FIELDS
    }
    if result["requests"] != result["hits"] + result["misses"]:
        raise DemoError(f"{name}: requests must equal hits + misses")
    if result["storage_bytes"] != result["storage_loads"] * EXPERT_BYTES:
        raise DemoError(f"{name}: logical storage byte/load invariant failed")
    if result["host_to_device_bytes"] != result["transfer_loads"] * EXPERT_BYTES:
        raise DemoError(f"{name}: logical H2D byte/load invariant failed")
    for field in ZERO_SAFETY_COUNTERS:
        if result[field] != 0:
            raise DemoError(f"{name}.{field} must be zero")
    return result


def _memory_payload(value: object, name: str) -> dict[str, int | None]:
    payload = _mapping(value, name)
    result: dict[str, int | None] = {}
    for field in ("rss_bytes", "peak_rss_bytes"):
        raw = payload.get(field)
        result[field] = None if raw is None else _integer(raw, f"{name}.{field}")
    return result


def _cuda_payload(value: object, name: str) -> dict[str, int]:
    payload = _mapping(value, name)
    return {
        field: _integer(payload.get(field), f"{name}.{field}")
        for field in (
            "baseline_allocated_bytes",
            "peak_allocated_bytes",
            "peak_incremental_bytes",
            "peak_reserved_bytes",
        )
    }


def validate_benchmark_result(
    payload: dict[str, Any],
    *,
    expected_mode: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != 1 or payload.get("status") != "ok":
        raise DemoError(f"{expected_mode} benchmark did not produce status=ok")
    evidence = _mapping(payload.get("evidence"), f"{expected_mode}.evidence")
    if evidence.get("publishable_benchmark_evidence") is not False:
        raise DemoError(
            f"{expected_mode} result must be explicitly marked as demo output"
        )
    model = _mapping(payload.get("model"), f"{expected_mode}.model")
    expected_model = _mapping(plan["model"], "plan.model")
    if model.get("model_id") != expected_model["id"]:
        raise DemoError(f"{expected_mode} benchmark model id does not match the plan")
    if model.get("revision") != expected_model["revision"]:
        raise DemoError(f"{expected_mode} benchmark revision does not match the plan")
    shards = _mapping(model.get("shards"), f"{expected_mode}.model.shards")
    expected_shards = _mapping(expected_model["shards_sha256"], "plan.model.shards")
    if set(shards) != set(expected_shards):
        raise DemoError(f"{expected_mode} benchmark shard set does not match the plan")
    for filename, expected_digest in expected_shards.items():
        shard = _mapping(shards[filename], f"{expected_mode}.model.shards.{filename}")
        if shard.get("sha256") != expected_digest:
            raise DemoError(f"{expected_mode} benchmark shard digest does not match")

    source = _mapping(payload.get("source"), f"{expected_mode}.source")
    expected_source = _mapping(plan["source"], "plan.source")
    if source.get("provenance_mode") != "demo" or source.get("tree_clean") is not None:
        raise DemoError(f"{expected_mode} result has non-demo source provenance")
    for field in ("benchmark_script_sha256", "paged_runtime_sha256"):
        if source.get(field) != expected_source[field]:
            raise DemoError(
                f"{expected_mode} benchmark source hash does not match the plan"
            )

    runtime = _mapping(payload.get("runtime"), f"{expected_mode}.runtime")
    budget = _mapping(runtime.get("budget"), f"{expected_mode}.runtime.budget")
    expected_runtime = _mapping(plan["runtime"], "plan.runtime")
    if runtime.get("pipeline") != expected_mode:
        raise DemoError(f"{expected_mode} benchmark pipeline does not match")
    if runtime.get("device") != expected_runtime["device"]:
        raise DemoError(f"{expected_mode} benchmark CUDA device does not match")
    device_uuid = runtime.get("device_uuid")
    if (
        not isinstance(device_uuid, str)
        or hashlib.sha256(device_uuid.encode("utf-8")).hexdigest()
        != expected_runtime["gpu_uuid_sha256"]
    ):
        raise DemoError(f"{expected_mode} benchmark CUDA UUID does not match")
    if runtime.get("device_name") != expected_runtime["gpu_name"]:
        raise DemoError(f"{expected_mode} benchmark CUDA name does not match")
    if runtime.get("policy") != "lru":
        raise DemoError(f"{expected_mode} benchmark policy must be lru")
    if _integer(budget.get("slots_per_layer"), "budget.slots_per_layer") != int(
        expected_runtime["slots_per_layer"]
    ):
        raise DemoError(f"{expected_mode} benchmark slot count does not match")
    if _integer(budget.get("staging_slots"), "budget.staging_slots") != 2:
        raise DemoError(f"{expected_mode} benchmark staging slot count must be two")
    expected_cache_bytes = int(expected_runtime["cache_bytes"])
    if (
        _integer(budget.get("cache_bytes"), "budget.cache_bytes")
        != expected_cache_bytes
    ):
        raise DemoError(f"{expected_mode} benchmark cache budget does not match")
    observed_total_vram = _integer(
        budget.get("device_total_vram_bytes"), "budget.device_total_vram_bytes"
    )
    if (
        abs(observed_total_vram - int(expected_runtime["gpu_total_vram_bytes"]))
        > 16 * 1024**2
    ):
        raise DemoError(f"{expected_mode} benchmark CUDA capacity does not match")
    if _integer(
        budget.get("non_expert_checkpoint_bytes"),
        "budget.non_expert_checkpoint_bytes",
    ) != int(expected_model["non_expert_bytes"]):
        raise DemoError(f"{expected_mode} benchmark non-expert budget does not match")

    workload = _mapping(payload.get("workload"), f"{expected_mode}.workload")
    expected_workload = _mapping(plan["workload"], "plan.workload")
    generation = _mapping(plan["generation"], "plan.generation")
    if workload.get("id") != expected_workload["id"]:
        raise DemoError(f"{expected_mode} benchmark workload id does not match")
    if workload.get("prompt_sha256") != expected_workload["prompt_sha256"]:
        raise DemoError(f"{expected_mode} benchmark prompt does not match")
    if _integer(workload.get("max_new_tokens"), "workload.max_new_tokens") != int(
        generation["max_new_tokens"]
    ):
        raise DemoError(f"{expected_mode} benchmark token limit does not match")
    input_tokens = _integer(
        workload.get("input_tokens"), "workload.input_tokens", minimum=1
    )
    if input_tokens > int(generation["max_input_tokens"]):
        raise DemoError(f"{expected_mode} benchmark input token limit does not match")
    if _integer(workload.get("seed"), "workload.seed") != int(generation["seed"]):
        raise DemoError(f"{expected_mode} benchmark seed does not match")

    model_load = _mapping(payload.get("model_load"), f"{expected_mode}.model_load")
    _number(model_load.get("total_seconds"), "model_load.total_seconds", positive=True)
    _cuda_payload(model_load.get("cuda_memory"), "model_load.cuda_memory")
    _memory_payload(
        model_load.get("process_memory_after"),
        "model_load.process_memory_after",
    )

    passes = _mapping(payload.get("passes"), f"{expected_mode}.passes")
    normalized: dict[str, Any] = {}
    for pass_name in PASS_NAMES:
        current = _mapping(passes.get(pass_name), f"{expected_mode}.{pass_name}")
        wall = _number(
            current.get("total_wall_seconds"),
            f"{expected_mode}.{pass_name}.total_wall_seconds",
            positive=True,
        )
        generated_count = _integer(
            current.get("generated_token_count"),
            f"{expected_mode}.{pass_name}.generated_token_count",
            minimum=1,
        )
        throughput = _number(
            current.get("end_to_end_generated_tokens_per_second_including_prefill"),
            f"{expected_mode}.{pass_name}.throughput",
            positive=True,
        )
        expected_throughput = generated_count / wall
        if not math.isclose(
            throughput, expected_throughput, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise DemoError(f"{expected_mode}.{pass_name} throughput is inconsistent")
        generated_ids = _array(
            current.get("generated_ids"), f"{expected_mode}.{pass_name}.generated_ids"
        )
        fed_ids = _array(
            current.get("fed_token_ids"), f"{expected_mode}.{pass_name}.fed_token_ids"
        )
        if len(generated_ids) != generated_count or len(fed_ids) != generated_count:
            raise DemoError(
                f"{expected_mode}.{pass_name} token arrays have wrong length"
            )
        for index, token in enumerate((*generated_ids, *fed_ids)):
            _integer(token, f"{expected_mode}.{pass_name}.token[{index}]")
        normalized[pass_name] = {
            "wall_seconds": wall,
            "generated_count": generated_count,
            "throughput": throughput,
            "generated_ids": generated_ids,
            "fed_ids": fed_ids,
            "generated_text": _string(
                current.get("generated_text"),
                f"{expected_mode}.{pass_name}.generated_text",
            ),
            "metrics": _validate_metrics(
                current.get("metrics"), f"{expected_mode}.{pass_name}.metrics"
            ),
            "cuda": _cuda_payload(
                current.get("cuda_memory"),
                f"{expected_mode}.{pass_name}.cuda_memory",
            ),
            "process": _memory_payload(
                current.get("process_memory_after"),
                f"{expected_mode}.{pass_name}.process_memory_after",
            ),
        }

    cold = normalized[PASS_NAMES[0]]
    retained = normalized[PASS_NAMES[1]]
    if cold["generated_ids"] != retained["generated_ids"]:
        raise DemoError(f"{expected_mode} cold/retained generated token IDs differ")
    if cold["fed_ids"] != retained["fed_ids"]:
        raise DemoError(f"{expected_mode} cold/retained fed token IDs differ")
    return normalized


def validate_mode_identity(validated: Mapping[str, dict[str, Any]]) -> None:
    if set(validated) != {"sync", "async"}:
        raise DemoError("comparison requires exactly one sync and one async result")
    for pass_name in PASS_NAMES:
        sync = validated["sync"][pass_name]
        async_ = validated["async"][pass_name]
        for field in ("generated_count", "generated_ids", "fed_ids", "metrics"):
            if sync[field] != async_[field]:
                raise DemoError(
                    f"sync/async identity gate failed at {pass_name}.{field}"
                )


def _max_optional(values: Sequence[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def build_summary(
    *,
    plan: dict[str, Any],
    reports: Mapping[str, dict[str, Any]],
    validated: Mapping[str, dict[str, Any]],
    artifact_paths: Mapping[str, Path],
    created_at: str | None = None,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in _array(plan["runtime"]["pipeline_modes"], "plan pipeline modes"):
        report = reports[mode]
        normalized = validated[mode]
        model_load = _mapping(report["model_load"], f"{mode}.model_load")
        load_cuda = _cuda_payload(model_load["cuda_memory"], "model_load.cuda")
        load_process = _memory_payload(
            model_load["process_memory_after"], "model_load.process"
        )
        cuda_rows = [load_cuda] + [normalized[name]["cuda"] for name in PASS_NAMES]
        process_rows = [load_process] + [
            normalized[name]["process"] for name in PASS_NAMES
        ]
        pass_summary = {
            name: {
                "wall_seconds": normalized[name]["wall_seconds"],
                "generated_tokens": normalized[name]["generated_count"],
                "tokens_per_second": normalized[name]["throughput"],
                "generated_token_ids": normalized[name]["generated_ids"],
                "peak_allocated_vram_bytes": normalized[name]["cuda"][
                    "peak_allocated_bytes"
                ],
                "peak_reserved_vram_bytes": normalized[name]["cuda"][
                    "peak_reserved_bytes"
                ],
                "process_rss_bytes": normalized[name]["process"]["rss_bytes"],
                "peak_process_working_set_bytes": normalized[name]["process"][
                    "peak_rss_bytes"
                ],
                "cache_hit_rate": (
                    0.0
                    if normalized[name]["metrics"]["requests"] == 0
                    else normalized[name]["metrics"]["hits"]
                    / normalized[name]["metrics"]["requests"]
                ),
                "logical_storage_bytes": normalized[name]["metrics"]["storage_bytes"],
                "logical_host_to_device_bytes": normalized[name]["metrics"][
                    "host_to_device_bytes"
                ],
                "generated_text": normalized[name]["generated_text"],
            }
            for name in PASS_NAMES
        }
        modes[mode] = {
            "model_load_seconds": _number(
                model_load["total_seconds"], "model_load.total_seconds", positive=True
            ),
            "passes": pass_summary,
            "memory": {
                "peak_allocated_vram_bytes": max(
                    row["peak_allocated_bytes"] for row in cuda_rows
                ),
                "peak_reserved_vram_bytes": max(
                    row["peak_reserved_bytes"] for row in cuda_rows
                ),
                "peak_process_working_set_bytes": _max_optional(
                    [row["peak_rss_bytes"] for row in process_rows]
                ),
                "final_process_rss_bytes": normalized[PASS_NAMES[1]]["process"][
                    "rss_bytes"
                ],
            },
        }

    comparison = None
    limitations = list(SUMMARY_LIMITATIONS)
    if set(modes) == {"sync", "async"}:
        comparison = {}
        for pass_name in PASS_NAMES:
            sync_wall = float(modes["sync"]["passes"][pass_name]["wall_seconds"])
            async_wall = float(modes["async"]["passes"][pass_name]["wall_seconds"])
            comparison[pass_name] = {
                "sync_wall_seconds": sync_wall,
                "async_wall_seconds": async_wall,
                "sync_over_async_ratio": sync_wall / async_wall,
                "saving_seconds": sync_wall - async_wall,
                "saving_fraction": (sync_wall - async_wall) / sync_wall,
            }
        limitations.append(
            "The optional sync/async comparison is one local pair and remains order-sensitive."
        )

    artifacts = {
        mode: {
            "path": path.name,
            "sha256": _sha256(path),
        }
        for mode, path in sorted(artifact_paths.items())
    }
    return {
        "schema_version": 1,
        "status": "ok",
        "evidence_label": "local one-command OLMoE paging demo",
        "publishable_benchmark_evidence": False,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "plan_id": plan["id"],
        "plan": plan,
        "cache_budget": {
            "slots_per_layer": plan["runtime"]["slots_per_layer"],
            "cache_bytes": plan["runtime"]["cache_bytes"],
            "staging_slots": plan["runtime"]["staging_slots"],
            "staging_host_bytes": 2 * EXPERT_BYTES,
            "non_expert_bytes": plan["model"]["non_expert_bytes"],
            "vram_reserve_bytes": plan["runtime"]["vram_reserve_bytes"],
        },
        "modes": modes,
        "comparison": comparison,
        "artifacts": artifacts,
        "limitations": limitations,
    }


def _validate_existing_summary(
    summary: dict[str, Any],
    *,
    plan: dict[str, Any],
    reports: Mapping[str, dict[str, Any]],
    validated: Mapping[str, dict[str, Any]],
    artifact_paths: Mapping[str, Path],
) -> None:
    created_at = _string(summary.get("created_at"), "summary.created_at")
    expected = build_summary(
        plan=plan,
        reports=reports,
        validated=validated,
        artifact_paths=artifact_paths,
        created_at=created_at,
    )
    if summary != expected:
        raise DemoError("existing summary.json does not match its validated artifacts")


def _run_or_resume_mode(
    config: DemoConfig,
    *,
    runner: ProcessRunner,
    mode: str,
    slots_per_layer: int,
    output: Path,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not output.exists():
        print(f"Running {mode} paged-runtime pass...")
        argv = benchmark_argv(
            config,
            mode=mode,
            slots_per_layer=slots_per_layer,
            output=output,
        )
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        result = runner.run(argv, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise DemoError(
                f"{mode} benchmark failed with exit code {result.returncode}: "
                f"{detail or 'no diagnostic output'}"
            )
        if not output.is_file():
            raise DemoError(f"{mode} benchmark succeeded without creating {output}")
    else:
        print(f"Reusing validated {mode} result from the matching demo plan.")
    report = _load_json(output, f"{mode} benchmark")
    normalized = validate_benchmark_result(report, expected_mode=mode, plan=plan)
    return report, normalized


def execute_demo(
    config: DemoConfig,
    *,
    model_id: str,
    revision: str,
    shard_sha256: Mapping[str, str],
    runner: ProcessRunner,
) -> tuple[dict[str, Any], Path]:
    if not config.snapshot.is_dir():
        raise DemoError(f"snapshot directory not found: {config.snapshot}")
    if not config.benchmark_script.is_file():
        raise DemoError(f"benchmark script not found: {config.benchmark_script}")
    if not config.python_executable.is_file():
        raise DemoError(f"Python executable not found: {config.python_executable}")
    if not 1 <= config.max_input_tokens <= 256:
        raise DemoError("max-input-tokens must be between 1 and 256")
    if not 1 <= config.max_new_tokens <= 64:
        raise DemoError("max-new-tokens must be between 1 and 64")

    checkpoint_bytes, non_expert_bytes = checkpoint_sizes(config.snapshot)
    gpu = probe_gpu(runner, config.device)
    slots, reserve_bytes, _affordable = plan_slots(
        gpu,
        non_expert_bytes,
        config.requested_slots,
    )
    plan = build_plan(
        config,
        model_id=model_id,
        revision=revision,
        shard_sha256=shard_sha256,
        gpu=gpu,
        slots_per_layer=slots,
        reserve_bytes=reserve_bytes,
        checkpoint_bytes=checkpoint_bytes,
        non_expert_bytes=non_expert_bytes,
    )
    run_dir = config.output_root / str(plan["id"])
    if config.dry_run:
        print_dry_run(plan, run_dir, gpu)
        return plan, run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        _validate_plan(_load_json(plan_path, "demo plan"), plan)
    else:
        _write_json_create_only(plan_path, plan)

    modes = [str(mode) for mode in plan["runtime"]["pipeline_modes"]]
    reports: dict[str, dict[str, Any]] = {}
    validated: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    for mode in modes:
        output = run_dir / f"{mode}.json"
        report, normalized = _run_or_resume_mode(
            config,
            runner=runner,
            mode=mode,
            slots_per_layer=slots,
            output=output,
            plan=plan,
        )
        reports[mode] = report
        validated[mode] = normalized
        artifact_paths[mode] = output
    if config.compare:
        validate_mode_identity(validated)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path, "demo summary")
        _validate_existing_summary(
            summary,
            plan=plan,
            reports=reports,
            validated=validated,
            artifact_paths=artifact_paths,
        )
    else:
        summary = build_summary(
            plan=plan,
            reports=reports,
            validated=validated,
            artifact_paths=artifact_paths,
        )
        _write_json_create_only(summary_path, summary)
    print_summary(summary, summary_path)
    return summary, summary_path


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1024**3:.2f} GiB"


def print_dry_run(plan: dict[str, Any], run_dir: Path, gpu: GpuInfo) -> None:
    runtime = plan["runtime"]
    print("MoEVM OLMoE demo - dry run")
    print(f"  GPU:          {gpu.name} ({_gib(gpu.free_bytes)} free)")
    print(f"  Pipeline:     {', '.join(runtime['pipeline_modes'])}")
    print(
        f"  GPU cache:    {runtime['slots_per_layer']} slots/layer ({_gib(runtime['cache_bytes'])})"
    )
    print(f"  Output:       {run_dir}")
    print("  No model benchmark or file write was performed.")


def print_summary(summary: dict[str, Any], summary_path: Path) -> None:
    print("\nMoEVM OLMoE demo completed")
    print(
        f"GPU cache: {summary['cache_budget']['slots_per_layer']} slots/layer "
        f"({_gib(summary['cache_budget']['cache_bytes'])})"
    )
    print()
    print(
        f"{'Mode':<9} {'Load':>8} {'Empty':>10} {'Retained':>10} "
        f"{'Empty tok/s':>13} {'Warm tok/s':>12} {'VRAM alloc':>12} "
        f"{'VRAM reserv':>12} {'RAM peak':>12}"
    )
    for mode, values in summary["modes"].items():
        cold = values["passes"][PASS_NAMES[0]]
        retained = values["passes"][PASS_NAMES[1]]
        memory = values["memory"]
        print(
            f"{mode:<9} {values['model_load_seconds']:>7.2f}s "
            f"{cold['wall_seconds']:>9.2f}s {retained['wall_seconds']:>9.2f}s "
            f"{cold['tokens_per_second']:>13.3f} "
            f"{retained['tokens_per_second']:>12.3f} "
            f"{_gib(memory['peak_allocated_vram_bytes']):>12} "
            f"{_gib(memory['peak_reserved_vram_bytes']):>12} "
            f"{_gib(memory['peak_process_working_set_bytes']):>12}"
        )
        print(
            " " * 11
            + "logical storage empty/retained: "
            + f"{_gib(cold['logical_storage_bytes'])} / "
            + f"{_gib(retained['logical_storage_bytes'])}; H2D: "
            + f"{_gib(cold['logical_host_to_device_bytes'])} / "
            + f"{_gib(retained['logical_host_to_device_bytes'])}"
        )
    if summary["comparison"] is not None:
        print("\nIllustrative sync vs async wall-time comparison:")
        for pass_name, values in summary["comparison"].items():
            async_change = -values["saving_fraction"] * 100.0
            print(
                f"  {pass_name}: {values['sync_wall_seconds']:.2f}s -> "
                f"{values['async_wall_seconds']:.2f}s "
                f"({async_change:+.1f}% async wall time)"
            )
    preferred_mode = (
        "async" if "async" in summary["modes"] else next(iter(summary["modes"]))
    )
    generated = summary["modes"][preferred_mode]["passes"][PASS_NAMES[0]]
    print(f"\nGenerated text ({preferred_mode}): {generated['generated_text']!r}")
    print(f"Generated token IDs: {generated['generated_token_ids']}")
    print(f"\nResult: {summary_path}")
    print("Note: short local smoke; not a production throughput claim.")


def _slots_argument(raw: str) -> str:
    if raw == "auto":
        return raw
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be 'auto' or an integer 1..32") from exc
    if not 1 <= value <= MAX_SLOTS_PER_LAYER:
        raise argparse.ArgumentTypeError("must be 'auto' or an integer 1..32")
    return str(value)


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
        type=Path,
        help="Verified pinned snapshot; bypasses cache discovery and download.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Hugging Face home used when --snapshot is omitted.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workload-file", type=Path, default=DEFAULT_WORKLOAD_FILE)
    parser.add_argument("--workload-id", default="systems_it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slots", type=_slots_argument, default="auto")
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run sync plus async and enforce exact token/counter identity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the deterministic plan without writing or running the model.",
    )
    network = parser.add_mutually_exclusive_group()
    network.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the pinned snapshot to be downloaded when absent.",
    )
    network.add_argument(
        "--offline",
        action="store_true",
        help="Require an already cached snapshot and refuse network fallback.",
    )
    return parser


def _load_assets() -> Any:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from moevm import olmoe_assets

    return olmoe_assets


def _resolve_snapshot(args: argparse.Namespace, assets: Any) -> Path:
    if args.snapshot is not None:
        snapshot = args.snapshot.expanduser().resolve()
        if args.allow_download:
            raise DemoError("--allow-download cannot be combined with --snapshot")
        if args.dry_run:
            if not snapshot.is_dir():
                raise DemoError(f"snapshot directory not found: {snapshot}")
            return snapshot
        assets.verify_pinned_snapshot(snapshot)
        return snapshot

    cache = args.cache.expanduser().resolve()
    if args.dry_run:
        snapshot = Path(assets.pinned_snapshot_path(cache)).resolve()
        if not snapshot.is_dir():
            raise DemoError(
                "dry-run requires an existing snapshot; no download was attempted"
            )
        return snapshot
    return Path(
        assets.ensure_pinned_snapshot(
            cache,
            allow_download=bool(args.allow_download and not args.offline),
        )
    ).resolve()


def main(
    argv: list[str] | None = None,
    *,
    runner: ProcessRunner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        assets = _load_assets()
        snapshot = _resolve_snapshot(args, assets)
        if not args.dry_run:
            print("Pinned OLMoE checkpoint verified; model execution is now offline.")
        config = DemoConfig(
            snapshot=snapshot,
            output_root=args.output_root.expanduser().resolve(),
            workload_file=args.workload_file.expanduser().resolve(),
            workload_id=args.workload_id,
            device=args.device,
            requested_slots=args.slots,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            compare=args.compare,
            dry_run=args.dry_run,
        )
        execute_demo(
            config,
            model_id=assets.PINNED_MODEL_ID,
            revision=assets.PINNED_REVISION,
            shard_sha256=assets.PINNED_SHARD_SHA256,
            runner=runner or SubprocessRunner(),
        )
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
