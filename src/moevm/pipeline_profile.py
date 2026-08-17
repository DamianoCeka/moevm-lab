"""Measured, hardware-bound pipeline selection profiles."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROFILE_KIND = "moevm-measured-pipeline-profile"
PROFILE_SCHEMA_VERSION = 1
PASS_NAMES = ("cold_expert_cache", "repeat_retained_expert_cache")
PIPELINE_PRIMITIVES = (
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


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _metric_scopes(value: object) -> dict[str, tuple[object, ...]]:
    scopes: dict[str, tuple[object, ...]] = {}

    def walk(item: object, path: str = "") -> None:
        if isinstance(item, Mapping):
            if all(field in item for field in ("requests", "hits", "misses")):
                scopes[path] = tuple(item.get(field) for field in PIPELINE_PRIMITIVES)
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value)
    return scopes


def result_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact fields that make a timing profile reusable."""
    model = _mapping(result.get("model"), "result.model")
    runtime = _mapping(result.get("runtime"), "result.runtime")
    budget = _mapping(runtime.get("budget"), "result.runtime.budget")
    workload = _mapping(result.get("workload"), "result.workload")
    source = _mapping(result.get("source"), "result.source")
    environment = _mapping(result.get("environment"), "result.environment")
    shards = _mapping(model.get("shards"), "result.model.shards")
    shard_hashes = {
        str(name): str(_mapping(info, f"shard {name}").get("sha256"))
        for name, info in sorted(shards.items())
    }
    return {
        "model": {
            "model_id": model.get("model_id"),
            "revision": model.get("revision"),
            "dtype": model.get("dtype"),
            "layers": model.get("layers"),
            "experts_per_layer": model.get("experts_per_layer"),
            "top_k": model.get("top_k"),
            "checkpoint_shards_sha256": shard_hashes,
        },
        "hardware": {
            "device_uuid": runtime.get("device_uuid"),
            "device_name": runtime.get("device_name"),
            "device_total_vram_bytes": budget.get("device_total_vram_bytes"),
        },
        "runtime": {
            "policy": runtime.get("policy"),
            "capacity_scope": runtime.get("capacity_scope"),
            "hotset_sha256": runtime.get("hotset_sha256"),
            "slots_per_layer": budget.get("slots_per_layer"),
            "layers": budget.get("layers"),
            "expert_bytes": budget.get("expert_bytes"),
            "cache_bytes": budget.get("cache_bytes"),
            "staging_slots": budget.get("staging_slots"),
            "staging_host_bytes": budget.get("staging_host_bytes"),
            "non_expert_checkpoint_bytes": budget.get("non_expert_checkpoint_bytes"),
            "expected_weight_vram_bytes": budget.get("expected_weight_vram_bytes"),
        },
        "workload": {
            "id": workload.get("id"),
            "prompt_sha256": workload.get("prompt_sha256"),
            "input_ids": workload.get("input_ids"),
            "input_tokens": workload.get("input_tokens"),
            "max_new_tokens": workload.get("max_new_tokens"),
            "decoding": workload.get("decoding"),
            "seed": workload.get("seed"),
        },
        "software": {
            "python": environment.get("python"),
            "platform": environment.get("platform"),
            "packages": environment.get("packages"),
            "provenance_mode": source.get("provenance_mode"),
            "benchmark_script_sha256": source.get("benchmark_script_sha256"),
            "paged_runtime_sha256": source.get("paged_runtime_sha256"),
        },
    }


def _validate_result(result: Mapping[str, Any], expected_pipeline: str) -> None:
    if result.get("schema_version") != 1 or result.get("status") != "ok":
        raise ValueError("calibration result must be a successful schema_version 1 run")
    runtime = _mapping(result.get("runtime"), "result.runtime")
    if runtime.get("pipeline") != expected_pipeline:
        raise ValueError(f"expected a {expected_pipeline} calibration result")
    evidence = _mapping(result.get("evidence"), "result.evidence")
    if evidence.get("publishable_benchmark_evidence") is not True:
        raise ValueError("calibration result must be benchmark evidence")
    source = _mapping(result.get("source"), "result.source")
    if source.get("tree_clean") is not True:
        raise ValueError("calibration result must come from a clean tree")
    reference = _mapping(result.get("reference_comparison"), "reference_comparison")
    if reference.get("available") is not True:
        raise ValueError("calibration result must include a reference gate")
    passes = _mapping(result.get("passes"), "result.passes")
    for pass_name in PASS_NAMES:
        measured_pass = _mapping(passes.get(pass_name), f"passes.{pass_name}")
        _positive_seconds(
            measured_pass.get("total_wall_seconds"),
            f"passes.{pass_name}.total_wall_seconds",
        )
    reference_mode = reference.get("mode")
    if reference_mode == "autoregressive_exact_gate":
        if reference.get("matched") is not True:
            raise ValueError("autoregressive calibration must match its reference")
    elif reference_mode == "teacher_forced":
        reference_ids = reference.get("generated_token_ids")
        if not isinstance(reference_ids, list) or not reference_ids:
            raise ValueError("teacher-forced reference token IDs are required")
        for pass_name in PASS_NAMES:
            measured_pass = _mapping(passes.get(pass_name), f"passes.{pass_name}")
            if measured_pass.get("teacher_forced") is not True:
                raise ValueError("teacher-forced calibration pass is not marked")
            if measured_pass.get("fed_token_ids") != reference_ids:
                raise ValueError(
                    "teacher-forced calibration must feed the exact reference IDs"
                )
    else:
        raise ValueError("unsupported calibration reference mode")


def _select_pipeline(ratios: Sequence[float], minimum_gain: float) -> tuple[str, str]:
    median_ratio = statistics.median(ratios)
    if all(ratio > 1.0 for ratio in ratios) and median_ratio >= 1.0 + minimum_gain:
        return "async", "async won every pair and cleared the minimum median gain"
    return (
        "sync",
        "sync is the fail-closed choice when async evidence is mixed or small",
    )


def build_measured_profile(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    minimum_gain: float = 0.03,
) -> dict[str, Any]:
    """Validate paired evidence and build a per-pass selection profile."""
    if len(pairs) < 3:
        raise ValueError("at least three paired sync/async runs are required")
    if not math.isfinite(minimum_gain) or not 0.0 <= minimum_gain <= 0.25:
        raise ValueError("minimum_gain must be between 0 and 0.25")

    canonical_binding: dict[str, Any] | None = None
    observations: dict[str, list[dict[str, Any]]] = {
        pass_name: [] for pass_name in PASS_NAMES
    }
    for pair_index, (sync_result, async_result) in enumerate(pairs, start=1):
        _validate_result(sync_result, "sync")
        _validate_result(async_result, "async")
        sync_binding = result_binding(sync_result)
        async_binding = result_binding(async_result)
        if sync_binding != async_binding:
            raise ValueError(f"pair {pair_index} has different bindings")
        if canonical_binding is None:
            canonical_binding = sync_binding
        elif sync_binding != canonical_binding:
            raise ValueError(f"pair {pair_index} does not match the first binding")
        if _metric_scopes(sync_result) != _metric_scopes(async_result):
            raise ValueError(
                f"pair {pair_index} has different cache/traffic primitives"
            )

        sync_passes = _mapping(sync_result.get("passes"), "sync passes")
        async_passes = _mapping(async_result.get("passes"), "async passes")
        for pass_name in PASS_NAMES:
            sync_pass = _mapping(sync_passes.get(pass_name), pass_name)
            async_pass = _mapping(async_passes.get(pass_name), pass_name)
            if sync_pass.get("generated_ids") != async_pass.get("generated_ids"):
                raise ValueError(f"pair {pair_index} generated token IDs differ")
            if sync_pass.get("fed_token_ids") != async_pass.get("fed_token_ids"):
                raise ValueError(f"pair {pair_index} fed token IDs differ")
            sync_seconds = _positive_seconds(
                sync_pass.get("total_wall_seconds"), "sync wall time"
            )
            async_seconds = _positive_seconds(
                async_pass.get("total_wall_seconds"), "async wall time"
            )
            observations[pass_name].append(
                {
                    "pair": pair_index,
                    "sync_seconds": sync_seconds,
                    "async_seconds": async_seconds,
                    "sync_over_async": sync_seconds / async_seconds,
                }
            )

    assert canonical_binding is not None
    selection: dict[str, str] = {}
    pass_evidence: dict[str, Any] = {}
    for pass_name in PASS_NAMES:
        ratios = [item["sync_over_async"] for item in observations[pass_name]]
        selected, reason = _select_pipeline(ratios, minimum_gain)
        selection[pass_name] = selected
        pass_evidence[pass_name] = {
            "selected": selected,
            "reason": reason,
            "paired_observations": observations[pass_name],
            "median_sync_over_async": statistics.median(ratios),
            "minimum_sync_over_async": min(ratios),
            "maximum_sync_over_async": max(ratios),
            "async_wins": sum(ratio > 1.0 for ratio in ratios),
        }

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "kind": PROFILE_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "selection": selection,
        "binding": canonical_binding,
        "calibration": {
            "pairs": len(pairs),
            "minimum_median_gain": minimum_gain,
            "policy": (
                "select async only when it wins every pair and its median paired "
                "speedup clears the threshold; otherwise select sync"
            ),
            "passes": pass_evidence,
        },
    }


def _validate_profile_shape(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("pipeline profile schema_version must be 1")
    if profile.get("kind") != PROFILE_KIND:
        raise ValueError(f"pipeline profile kind must be {PROFILE_KIND}")
    selection = _mapping(profile.get("selection"), "profile.selection")
    if set(selection) != set(PASS_NAMES):
        raise ValueError("pipeline profile selection must contain both pass names")
    for pass_name in PASS_NAMES:
        if selection.get(pass_name) not in ("sync", "async"):
            raise ValueError(f"invalid selection for {pass_name}")
    _mapping(profile.get("binding"), "profile.binding")
    calibration = _mapping(profile.get("calibration"), "profile.calibration")
    pairs = calibration.get("pairs")
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 3:
        raise ValueError("pipeline profile must contain at least three pairs")


def load_pipeline_profile(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.expanduser().read_bytes()
    try:
        profile = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pipeline profile must be valid UTF-8 JSON") from exc
    if not isinstance(profile, dict):
        raise ValueError("pipeline profile must be a JSON object")
    _validate_profile_shape(profile)
    return profile, hashlib.sha256(raw).hexdigest()


def _first_difference(expected: object, actual: object, path: str = "binding") -> str:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            return f"{path} keys"
        for key in sorted(expected):
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if expected != actual:
        return path
    return ""


def validate_profile_binding(
    profile: Mapping[str, Any], expected_binding: Mapping[str, Any]
) -> None:
    _validate_profile_shape(profile)
    actual_binding = _mapping(profile.get("binding"), "profile.binding")
    difference = _first_difference(expected_binding, actual_binding)
    if difference:
        raise ValueError(f"pipeline profile does not match current run at {difference}")
