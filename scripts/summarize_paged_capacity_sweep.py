#!/usr/bin/env python3
"""Validate and sanitize a real OLMoE paged-capacity sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

PINNED_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PINNED_REVISION = "bd1c52f59153f724c1ad11ca1791edc77bab3806"
PINNED_LAYERS = 16
PINNED_EXPERT_BYTES = 12_582_912
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


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verify_clean_source(repo_root: Path, expected_commit: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError("source_commit does not match the checked-out HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("refusing to summarize evidence from a dirty worktree")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _token_ids(value: object, name: str, expected_count: int) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or any(isinstance(token, bool) or not isinstance(token, int) for token in value)
    ):
        raise ValueError(f"{name} must contain exactly {expected_count} integers")
    return value


def _workloads(path: Path) -> list[dict[str, str]]:
    payload = _load_json(path)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("workloads"), list
    ):
        raise ValueError("workload file must be a schema_version 1 collection")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in payload["workloads"]:
        if not isinstance(item, dict):
            raise ValueError("workload entries must be objects")
        workload_id = item.get("id")
        prompt = item.get("prompt")
        if (
            not isinstance(workload_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]+", workload_id) is None
            or not isinstance(prompt, str)
            or not prompt
        ):
            raise ValueError("workload entries require a safe id and non-empty prompt")
        if workload_id in seen:
            raise ValueError(f"duplicate workload id: {workload_id}")
        seen.add(workload_id)
        rows.append({"id": workload_id, "prompt": prompt})
    if not rows:
        raise ValueError("workload collection cannot be empty")
    return rows


def _validate_baseline(
    path: Path,
    *,
    workload: dict[str, str],
    max_new_tokens: int,
    seed: int,
    workload_file_sha256: str,
) -> tuple[dict[str, Any], list[int]]:
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"baseline schema mismatch: {path}")
    model = payload.get("model")
    generation = payload.get("generation")
    workload_payload = payload.get("workload")
    timing = payload.get("timing_observation")
    environment = payload.get("environment")
    if not all(
        isinstance(item, dict)
        for item in (model, generation, workload_payload, timing, environment)
    ):
        raise ValueError(f"baseline sections missing: {path}")
    if (
        model.get("id") != PINNED_MODEL_ID
        or model.get("requested_revision") != PINNED_REVISION
        or model.get("resolved_revision") != PINNED_REVISION
        or model.get("checkpoint_shards_sha256") != PINNED_SHARD_SHA256
    ):
        raise ValueError(f"baseline model identity mismatch: {path}")
    prompt_sha256 = hashlib.sha256(workload["prompt"].encode()).hexdigest()
    if (
        workload_payload.get("id") != workload["id"]
        or workload_payload.get("prompt_sha256") != prompt_sha256
        or workload_payload.get("workload_file_sha256") != workload_file_sha256
    ):
        raise ValueError(f"baseline workload identity mismatch: {path}")
    if (
        _number(generation.get("temperature"), "baseline temperature") != 0.0
        or generation.get("seed") != seed
        or generation.get("generated_tokens") != max_new_tokens
    ):
        raise ValueError(f"baseline must be greedy: {path}")
    ids = _token_ids(
        generation.get("generated_token_ids"),
        "baseline generated_token_ids",
        max_new_tokens,
    )
    summary = {
        "wall_seconds": _number(
            timing.get("generation_wall_seconds"),
            "baseline generation_wall_seconds",
            minimum=1e-12,
        ),
        "prefill_seconds": _number(
            timing.get("prefill_seconds"), "baseline prefill_seconds"
        ),
        "decode_seconds": _number(
            timing.get("generation_decode_seconds"),
            "baseline generation_decode_seconds",
        ),
        "reference_token_ids": ids,
        "peak_vram_bytes": _integer(
            environment.get("peak_vram_bytes"), "baseline peak_vram_bytes", minimum=1
        ),
        "raw_sha256": _sha256(path),
    }
    return summary, ids


def _pass_summary(
    payload: dict[str, Any],
    *,
    label: str,
    reference_ids: list[int],
    expert_bytes: int,
) -> dict[str, Any]:
    metrics = payload.get("metrics")
    decode = payload.get("decode")
    prefill = payload.get("prefill")
    cuda_memory = payload.get("cuda_memory")
    process_memory = payload.get("process_memory_after")
    if not all(
        isinstance(item, dict)
        for item in (metrics, decode, prefill, cuda_memory, process_memory)
    ):
        raise ValueError(f"paged {label} sections missing")
    generated = _token_ids(
        payload.get("generated_ids"), f"paged {label} generated_ids", len(reference_ids)
    )
    fed = _token_ids(
        payload.get("fed_token_ids"), f"paged {label} fed_token_ids", len(reference_ids)
    )
    if not payload.get("teacher_forced") or fed != reference_ids:
        raise ValueError(f"paged {label} did not use the exact reference sequence")
    if payload.get("generated_token_count") != len(reference_ids):
        raise ValueError(f"paged {label} generated token count mismatch")
    requests = _integer(metrics.get("requests"), f"paged {label} requests", minimum=1)
    hits = _integer(metrics.get("hits"), f"paged {label} hits")
    misses = _integer(metrics.get("misses"), f"paged {label} misses")
    evictions = _integer(metrics.get("evictions"), f"paged {label} evictions")
    if hits + misses != requests:
        raise ValueError(f"paged {label} request counters are inconsistent")
    storage_bytes = _integer(
        metrics.get("storage_bytes"), f"paged {label} storage_bytes"
    )
    host_to_device_bytes = _integer(
        metrics.get("host_to_device_bytes"),
        f"paged {label} host_to_device_bytes",
    )
    if storage_bytes != misses * expert_bytes or host_to_device_bytes != storage_bytes:
        raise ValueError(f"paged {label} transfer counters are inconsistent")
    decode_tokens = _integer(
        decode.get("token_count"), f"paged {label} decode token_count"
    )
    if decode_tokens != len(reference_ids) - 1:
        raise ValueError(f"paged {label} decode token count mismatch")
    return {
        "wall_seconds": _number(
            payload.get("total_wall_seconds"),
            f"paged {label} total_wall_seconds",
            minimum=1e-12,
        ),
        "prefill_seconds": _number(
            prefill.get("wall_seconds"), f"paged {label} prefill wall_seconds"
        ),
        "decode_seconds": _number(
            decode.get("wall_seconds"),
            f"paged {label} decode wall_seconds",
            minimum=1e-12,
        ),
        "decode_tokens": decode_tokens,
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "logical_storage_bytes": storage_bytes,
        "host_to_device_bytes": host_to_device_bytes,
        "storage_seconds": _number(
            metrics.get("storage_seconds"), f"paged {label} storage_seconds"
        ),
        "transfer_seconds": _number(
            metrics.get("transfer_seconds"), f"paged {label} transfer_seconds"
        ),
        "forward_seconds": _number(
            metrics.get("forward_seconds"), f"paged {label} forward_seconds"
        ),
        "peak_vram_bytes": _integer(
            cuda_memory.get("peak_allocated_bytes"),
            f"paged {label} peak_allocated_bytes",
            minimum=1,
        ),
        "peak_reserved_vram_bytes": _integer(
            cuda_memory.get("peak_reserved_bytes"),
            f"paged {label} peak_reserved_bytes",
            minimum=1,
        ),
        "peak_rss_bytes": _integer(
            process_memory.get("peak_rss_bytes"),
            f"paged {label} peak_rss_bytes",
            minimum=1,
        ),
        "prediction_matches": sum(
            actual == expected
            for actual, expected in zip(generated, reference_ids, strict=True)
        ),
        "prediction_total": len(reference_ids),
        "generated_token_ids": generated,
        "fed_token_ids": fed,
    }


def _validate_paged(
    path: Path,
    *,
    workload: dict[str, str],
    capacity: int,
    max_new_tokens: int,
    seed: int,
    reference_ids: list[int],
    baseline_sha256: str,
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != 1 or payload.get("status") != "ok":
        raise ValueError(f"paged result is not schema_version 1 status ok: {path}")
    model = payload.get("model")
    runtime = payload.get("runtime")
    workload_payload = payload.get("workload")
    passes = payload.get("passes")
    reference = payload.get("reference_comparison")
    environment = payload.get("environment")
    if not all(
        isinstance(item, dict)
        for item in (
            model,
            runtime,
            workload_payload,
            passes,
            reference,
            environment,
        )
    ):
        raise ValueError(f"paged sections missing: {path}")
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("revision") != PINNED_REVISION
    ):
        raise ValueError(f"paged model identity mismatch: {path}")
    shards = model.get("shards")
    if (
        not isinstance(shards, dict)
        or {
            name: item.get("sha256") if isinstance(item, dict) else None
            for name, item in shards.items()
        }
        != PINNED_SHARD_SHA256
    ):
        raise ValueError(f"paged shard verification mismatch: {path}")
    budget = runtime.get("budget")
    if not isinstance(budget, dict):
        raise ValueError(f"paged runtime budget missing: {path}")
    expert_bytes = _integer(budget.get("expert_bytes"), "paged expert_bytes", minimum=1)
    layers = _integer(budget.get("layers"), "paged layers", minimum=1)
    cache_bytes = _integer(budget.get("cache_bytes"), "paged cache_bytes", minimum=1)
    non_expert_bytes = _integer(
        budget.get("non_expert_checkpoint_bytes"),
        "paged non_expert_checkpoint_bytes",
        minimum=1,
    )
    expected_weight_vram_bytes = _integer(
        budget.get("expected_weight_vram_bytes"),
        "paged expected_weight_vram_bytes",
        minimum=1,
    )
    if (
        layers != PINNED_LAYERS
        or expert_bytes != PINNED_EXPERT_BYTES
        or cache_bytes != layers * capacity * expert_bytes
        or expected_weight_vram_bytes != cache_bytes + non_expert_bytes
        or budget.get("slots_per_layer") != capacity
        or budget.get("staging_slots") != 1
    ):
        raise ValueError(f"paged runtime budget mismatch: {path}")
    prompt_sha256 = hashlib.sha256(workload["prompt"].encode()).hexdigest()
    if (
        runtime.get("policy") != "lru"
        or workload_payload.get("id") != workload["id"]
        or workload_payload.get("prompt_sha256") != prompt_sha256
        or workload_payload.get("max_new_tokens") != max_new_tokens
        or workload_payload.get("seed") != seed
        or workload_payload.get("decoding")
        != "teacher-forced reference with greedy predictions"
    ):
        raise ValueError(f"paged protocol mismatch: {path}")
    if (
        reference.get("available") is not True
        or reference.get("mode") != "teacher_forced"
        or reference.get("sha256") != baseline_sha256
        or reference.get("generated_token_ids") != reference_ids
    ):
        raise ValueError(f"paged reference binding mismatch: {path}")
    cold = _pass_summary(
        passes.get("cold_expert_cache", {}),
        label="cold",
        reference_ids=reference_ids,
        expert_bytes=expert_bytes,
    )
    retained = _pass_summary(
        passes.get("repeat_retained_expert_cache", {}),
        label="retained",
        reference_ids=reference_ids,
        expert_bytes=expert_bytes,
    )
    if cold["generated_token_ids"] != retained["generated_token_ids"]:
        raise ValueError(f"paged cold and retained predictions differ: {path}")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(f"paged created_at missing: {path}")
    device_name = runtime.get("device_name")
    python_version = environment.get("python")
    platform_name = environment.get("platform")
    packages = environment.get("packages")
    if (
        not isinstance(device_name, str)
        or not device_name
        or not isinstance(python_version, str)
        or not python_version
        or not isinstance(platform_name, str)
        or not platform_name
        or not isinstance(packages, dict)
        or not packages
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in packages.items()
        )
    ):
        raise ValueError(f"paged hardware/software metadata missing: {path}")
    return {
        "created_at": created_at,
        "input_tokens": _integer(
            workload_payload.get("input_tokens"), "paged input_tokens", minimum=1
        ),
        "budget": {
            "cache_bytes": cache_bytes,
            "expert_bytes": expert_bytes,
            "layers": layers,
            "non_expert_checkpoint_bytes": non_expert_bytes,
            "expected_weight_vram_bytes": expected_weight_vram_bytes,
        },
        "software": {
            "python": python_version,
            "platform": platform_name,
            "packages": packages,
        },
        "device_name": device_name,
        "cold": cold,
        "retained": retained,
        "raw_sha256": _sha256(path),
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    baseline_wall_seconds: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("cold", "retained"):
        passes = [row[label] for row in rows]
        wall_seconds = sum(item["wall_seconds"] for item in passes)
        decode_seconds = sum(item["decode_seconds"] for item in passes)
        decode_tokens = sum(item["decode_tokens"] for item in passes)
        requests = sum(item["requests"] for item in passes)
        hits = sum(item["hits"] for item in passes)
        prediction_matches = sum(item["prediction_matches"] for item in passes)
        prediction_total = sum(item["prediction_total"] for item in passes)
        result[label] = {
            "wall_seconds": wall_seconds,
            "speedup_over_baseline": baseline_wall_seconds / wall_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_tokens / decode_seconds,
            "requests": requests,
            "hits": hits,
            "misses": sum(item["misses"] for item in passes),
            "evictions": sum(item["evictions"] for item in passes),
            "hit_rate": hits / requests,
            "logical_storage_bytes": sum(
                item["logical_storage_bytes"] for item in passes
            ),
            "host_to_device_bytes": sum(
                item["host_to_device_bytes"] for item in passes
            ),
            "storage_seconds": sum(item["storage_seconds"] for item in passes),
            "transfer_seconds": sum(item["transfer_seconds"] for item in passes),
            "forward_seconds": sum(item["forward_seconds"] for item in passes),
            "peak_vram_bytes": max(item["peak_vram_bytes"] for item in passes),
            "peak_reserved_vram_bytes": max(
                item["peak_reserved_vram_bytes"] for item in passes
            ),
            "peak_rss_bytes": max(item["peak_rss_bytes"] for item in passes),
            "prediction_matches": prediction_matches,
            "prediction_total": prediction_total,
            "prediction_match_rate": prediction_matches / prediction_total,
        }
    return result


def _aggregate_repetitions(
    repetitions: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    metrics = [repetition["aggregate"][label] for repetition in repetitions]
    wall_seconds = [item["wall_seconds"] for item in metrics]
    speedups = [item["speedup_over_baseline"] for item in metrics]
    return {
        "repetitions": len(metrics),
        "wall_seconds_median": statistics.median(wall_seconds),
        "wall_seconds_min": min(wall_seconds),
        "wall_seconds_max": max(wall_seconds),
        "speedup_over_baseline_median": statistics.median(speedups),
        "speedup_over_baseline_min": min(speedups),
        "speedup_over_baseline_max": max(speedups),
        "decode_tokens_per_second_median": statistics.median(
            item["decode_tokens_per_second"] for item in metrics
        ),
        "requests": sum(item["requests"] for item in metrics),
        "hits": sum(item["hits"] for item in metrics),
        "misses": sum(item["misses"] for item in metrics),
        "evictions": sum(item["evictions"] for item in metrics),
        "hit_rate": sum(item["hits"] for item in metrics)
        / sum(item["requests"] for item in metrics),
        "logical_storage_bytes": sum(item["logical_storage_bytes"] for item in metrics),
        "host_to_device_bytes": sum(item["host_to_device_bytes"] for item in metrics),
        "peak_vram_bytes": max(item["peak_vram_bytes"] for item in metrics),
        "peak_reserved_vram_bytes": max(
            item["peak_reserved_vram_bytes"] for item in metrics
        ),
        "peak_rss_bytes": max(item["peak_rss_bytes"] for item in metrics),
        "prediction_matches": sum(item["prediction_matches"] for item in metrics),
        "prediction_total": sum(item["prediction_total"] for item in metrics),
        "prediction_match_rate": sum(item["prediction_matches"] for item in metrics)
        / sum(item["prediction_total"] for item in metrics),
    }


def _pareto(capacities: list[dict[str, Any]], label: str) -> list[int]:
    frontier: list[int] = []
    for candidate in capacities:
        candidate_metrics = candidate["aggregate_across_repetitions"][label]
        dominated = any(
            other is not candidate
            and other["aggregate_across_repetitions"][label]["peak_vram_bytes"]
            <= candidate_metrics["peak_vram_bytes"]
            and other["aggregate_across_repetitions"][label]["wall_seconds_median"]
            <= candidate_metrics["wall_seconds_median"]
            and (
                other["aggregate_across_repetitions"][label]["peak_vram_bytes"]
                < candidate_metrics["peak_vram_bytes"]
                or other["aggregate_across_repetitions"][label]["wall_seconds_median"]
                < candidate_metrics["wall_seconds_median"]
            )
            for other in capacities
        )
        if not dominated:
            frontier.append(candidate["slots_per_layer"])
    return frontier


def summarize_capacity_sweep(
    *,
    workload_file: Path,
    baseline_dir: Path,
    paged_dir: Path,
    capacities: tuple[int, ...],
    repetitions: int,
    max_new_tokens: int,
    seed: int,
    source_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git SHA")
    if (
        not capacities
        or len(set(capacities)) != len(capacities)
        or any(capacity <= 0 for capacity in capacities)
    ):
        raise ValueError("capacities must be unique positive integers")
    if repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if not 2 <= max_new_tokens <= 64:
        raise ValueError("max_new_tokens must be between 2 and 64")
    workloads = _workloads(workload_file)
    workload_file_sha256 = _sha256(workload_file)
    baseline_rows: list[dict[str, Any]] = []
    reference_ids: dict[str, list[int]] = {}
    for workload in workloads:
        baseline_path = baseline_dir / f"{workload['id']}.metadata.json"
        summary, ids = _validate_baseline(
            baseline_path,
            workload=workload,
            max_new_tokens=max_new_tokens,
            seed=seed,
            workload_file_sha256=workload_file_sha256,
        )
        baseline_rows.append({"workload": workload["id"], **summary})
        reference_ids[workload["id"]] = ids
    baseline_wall_seconds = sum(row["wall_seconds"] for row in baseline_rows)

    capacity_rows: list[dict[str, Any]] = []
    for capacity in capacities:
        repetition_rows: list[dict[str, Any]] = []
        for repetition in range(1, repetitions + 1):
            workload_rows: list[dict[str, Any]] = []
            for workload in workloads:
                baseline_row = next(
                    row for row in baseline_rows if row["workload"] == workload["id"]
                )
                paged_path = (
                    paged_dir
                    / f"repetition-{repetition}"
                    / f"slots-{capacity}"
                    / f"{workload['id']}.json"
                )
                paged = _validate_paged(
                    paged_path,
                    workload=workload,
                    capacity=capacity,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    reference_ids=reference_ids[workload["id"]],
                    baseline_sha256=baseline_row["raw_sha256"],
                )
                workload_rows.append(
                    {"workload": workload["id"], "repetition": repetition, **paged}
                )
            aggregate = _aggregate(
                workload_rows,
                baseline_wall_seconds=baseline_wall_seconds,
            )
            repetition_rows.append(
                {
                    "repetition": repetition,
                    "workloads": workload_rows,
                    "aggregate": aggregate,
                    "retained_over_cold": {
                        "wall_speedup": aggregate["cold"]["wall_seconds"]
                        / aggregate["retained"]["wall_seconds"],
                        "logical_storage_traffic_reduction": (
                            1.0
                            - aggregate["retained"]["logical_storage_bytes"]
                            / aggregate["cold"]["logical_storage_bytes"]
                            if aggregate["cold"]["logical_storage_bytes"] > 0
                            else None
                        ),
                    },
                }
            )
        observations = [
            workload
            for repetition_row in repetition_rows
            for workload in repetition_row["workloads"]
        ]
        budgets = [row["budget"] for row in observations]
        if any(budget != budgets[0] for budget in budgets[1:]):
            raise ValueError(f"capacity {capacity} has inconsistent runtime budgets")
        capacity_rows.append(
            {
                "slots_per_layer": capacity,
                "cache_bytes": budgets[0]["cache_bytes"],
                "expected_weight_vram_bytes": budgets[0]["expected_weight_vram_bytes"],
                "repetitions": repetition_rows,
                "aggregate_across_repetitions": {
                    label: _aggregate_repetitions(repetition_rows, label=label)
                    for label in ("cold", "retained")
                },
            }
        )

    fastest_cold = min(
        row["aggregate_across_repetitions"]["cold"]["wall_seconds_median"]
        for row in capacity_rows
    )
    eligible = [
        row
        for row in capacity_rows
        if row["aggregate_across_repetitions"]["cold"]["wall_seconds_median"]
        <= fastest_cold * 1.05
        and row["aggregate_across_repetitions"]["cold"]["speedup_over_baseline_median"]
        > 1.0
    ]
    balanced = min(eligible, key=lambda row: row["slots_per_layer"], default=None)
    execution_order = sorted(
        (
            {
                "created_at": observation["created_at"],
                "slots_per_layer": capacity["slots_per_layer"],
                "workload": observation["workload"],
                "repetition": observation["repetition"],
            }
            for capacity in capacity_rows
            for repetition_row in capacity["repetitions"]
            for observation in repetition_row["workloads"]
        ),
        key=lambda item: item["created_at"],
    )
    first_observation = capacity_rows[0]["repetitions"][0]["workloads"][0]
    all_observations = [
        observation
        for capacity in capacity_rows
        for repetition_row in capacity["repetitions"]
        for observation in repetition_row["workloads"]
    ]
    if any(
        observation["device_name"] != first_observation["device_name"]
        or observation["software"] != first_observation["software"]
        for observation in all_observations[1:]
    ):
        raise ValueError(
            "capacity sweep mixes different hardware/software environments"
        )
    raw_artifacts_sha256 = {
        **{f"baseline/{row['workload']}": row["raw_sha256"] for row in baseline_rows},
        **{
            f"repetition-{row['repetition']}/slots-{capacity['slots_per_layer']}/{row['workload']}": row[
                "raw_sha256"
            ]
            for capacity in capacity_rows
            for repetition_row in capacity["repetitions"]
            for row in repetition_row["workloads"]
        },
    }
    return {
        "schema_version": 1,
        "evidence_class": "end-to-end",
        "evidence_label": "real OLMoE paged expert-cache capacity sweep",
        "source": {
            "commit": source_commit,
            "worktree_clean": True,
            "benchmark_harness_sha256": _sha256(
                Path(__file__).with_name("benchmark_paged_olmoe.py")
            ),
            "summary_harness_sha256": _sha256(Path(__file__)),
        },
        "model": {"id": PINNED_MODEL_ID, "revision": PINNED_REVISION},
        "hardware": {"device_name": first_observation["device_name"]},
        "software": first_observation["software"],
        "protocol": {
            "workloads": len(workloads),
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "repetitions": repetitions,
            "teacher_forced": True,
            "capacities_slots_per_layer": list(capacities),
            "capacity_bytes_per_slot_per_layer": PINNED_EXPERT_BYTES,
            "workload_file_sha256": workload_file_sha256,
            "execution_order": execution_order,
            "balanced_selection_rule": (
                "smallest capacity within 5% of the fastest aggregate cold wall "
                "time that also beats the baseline"
            ),
        },
        "baseline": {
            "wall_seconds": baseline_wall_seconds,
            "decode_seconds": sum(row["decode_seconds"] for row in baseline_rows),
            "peak_vram_bytes": max(row["peak_vram_bytes"] for row in baseline_rows),
            "workloads": baseline_rows,
        },
        "capacities": capacity_rows,
        "pareto": {
            "cold_slots_per_layer": _pareto(capacity_rows, "cold"),
            "retained_slots_per_layer": _pareto(capacity_rows, "retained"),
        },
        "balanced_selection": (
            {
                "eligible": True,
                "slots_per_layer": balanced["slots_per_layer"],
                "cold_wall_seconds_median": balanced["aggregate_across_repetitions"][
                    "cold"
                ]["wall_seconds_median"],
                "cold_speedup_over_baseline_median": balanced[
                    "aggregate_across_repetitions"
                ]["cold"]["speedup_over_baseline_median"],
                "cold_wall_seconds_range": [
                    balanced["aggregate_across_repetitions"]["cold"][
                        "wall_seconds_min"
                    ],
                    balanced["aggregate_across_repetitions"]["cold"][
                        "wall_seconds_max"
                    ],
                ],
            }
            if balanced is not None
            else {
                "eligible": False,
                "reason": "no tested capacity met the predeclared selection rule",
            }
        ),
        "raw_artifacts_sha256": raw_artifacts_sha256,
        "limitations": [
            "Accelerate CPU offload is not a tuned production serving baseline.",
            "The secondary Accelerate comparison uses one baseline pass per workload, while paged capacities use repeated measurements.",
            "Teacher forcing fixes continuations for timing and does not establish autoregressive identity.",
            "Cold means an empty dynamic expert cache, not a cold OS cache or NVMe device.",
            "Storage counters are logical requested bytes, not physical NVMe telemetry.",
            f"{repetitions} repetitions provide only a limited estimate of run-to-run variance.",
            "The synchronous Python runtime does not overlap storage, transfer and compute.",
        ],
    }


def _parse_capacities(raw: str) -> tuple[int, ...]:
    try:
        capacities = tuple(int(item) for item in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "capacities must be comma-separated integers"
        ) from exc
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise argparse.ArgumentTypeError("capacities must be positive")
    return capacities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-file", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--paged-dir", required=True)
    parser.add_argument(
        "--capacities", type=_parse_capacities, default=(16, 24, 32, 40)
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 2 <= args.max_new_tokens <= 64:
        raise ValueError("max-new-tokens must be between 2 and 64")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    repo_root = Path(__file__).resolve().parent.parent
    _verify_clean_source(repo_root, args.source_commit)
    report = summarize_capacity_sweep(
        workload_file=Path(args.workload_file),
        baseline_dir=Path(args.baseline_dir),
        paged_dir=Path(args.paged_dir),
        capacities=args.capacities,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        source_commit=args.source_commit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
