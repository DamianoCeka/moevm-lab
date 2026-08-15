#!/usr/bin/env python3
"""Validate and sanitize the paired RTX 6000 Ada paged-runtime study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPARATOR_PATH = ROOT / "scripts" / "compare_paged_pipeline_pair.py"
PASS_LABELS = {
    "cold_expert_cache": "cold",
    "repeat_retained_expert_cache": "retained",
}
PAIR_ORDERS = {1: "async-sync", 2: "sync-async", 3: "async-sync"}
CORE_WORKLOADS = (
    "domain_switch",
    "math_reasoning",
    "python_code",
    "systems_en",
    "systems_it",
)
EXPECTED_CASES = frozenset(
    {
        *(
            ("core", workload, 16, 32, repetition)
            for workload in CORE_WORKLOADS
            for repetition in range(1, 4)
        ),
        *(
            ("length", "python_code", tokens, 32, repetition)
            for tokens in (2, 8, 32, 64)
            for repetition in range(1, 4)
        ),
        *(
            ("capacity", "python_code", 16, slots, repetition)
            for slots in (16, 24, 40)
            for repetition in range(1, 4)
        ),
    }
)


def _load_comparator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "paged_pair_comparator", COMPARATOR_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - repository invariant
        raise RuntimeError(f"cannot import pair comparator: {COMPARATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _median(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise ValueError("cannot aggregate an empty series")
    return statistics.median(collected)


def _mad(values: Iterable[float]) -> float:
    collected = list(values)
    center = _median(collected)
    return _median(abs(value - center) for value in collected)


def _case_from_path(
    study_root: Path, pair_path: Path
) -> tuple[str, str, int, int, int]:
    parts = pair_path.relative_to(study_root).parts
    if len(parts) != 6 or parts[-1] != "pair.json":
        raise ValueError(f"unexpected pair path: {pair_path}")
    group, workload, tokens_part, slots_part, repetition_part, _ = parts
    try:
        tokens = int(tokens_part.removeprefix("tokens-"))
        slots = int(slots_part.removeprefix("slots-"))
        repetition = int(repetition_part.removeprefix("repetition-"))
    except ValueError as exc:
        raise ValueError(f"invalid dimensions in pair path: {pair_path}") from exc
    if tokens_part != f"tokens-{tokens}" or slots_part != f"slots-{slots}":
        raise ValueError(f"non-canonical dimensions in pair path: {pair_path}")
    if repetition_part != f"repetition-{repetition}":
        raise ValueError(f"non-canonical repetition in pair path: {pair_path}")
    return group, workload, tokens, slots, repetition


def _safe_model(report: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(_mapping(report.get("model"), "model"))
    model.pop("snapshot", None)
    model.pop("hash_verification_seconds", None)
    model["dtype"] = str(model.get("dtype", "")).removeprefix("torch.")
    return model


def _safe_environment(report: Mapping[str, Any]) -> dict[str, Any]:
    environment = _mapping(report.get("environment"), "environment")
    packages = _mapping(environment.get("packages"), "environment.packages")
    return {
        "gpu": _mapping(report.get("runtime"), "runtime").get("device_name"),
        "platform": environment.get("platform"),
        "python": environment.get("python"),
        "packages": dict(packages),
    }


def _validate_case(
    study_root: Path,
    pair_path: Path,
    case: tuple[str, str, int, int, int],
    comparator: Any,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    group, workload, tokens, slots, repetition = case
    pair = _load_json(pair_path)
    sync_path = pair_path.with_name("sync.json")
    async_path = pair_path.with_name("async.json")
    sync = _load_json(sync_path)
    async_ = _load_json(async_path)

    pair_inputs = _mapping(pair.get("inputs"), "pair.inputs")
    for mode, path in (("sync", sync_path), ("async", async_path)):
        identity = _mapping(pair_inputs.get(mode), f"pair.inputs.{mode}")
        if identity.get("sha256") != _sha256(path):
            raise ValueError(f"{mode} SHA-256 mismatch for {pair_path}")

    recomputed = comparator.compare_reports(sync, async_)
    for field in (
        "schema_version",
        "status",
        "source_commit",
        "benchmark_script_sha256",
        "exact_invariants",
        "passes",
    ):
        if pair.get(field) != recomputed.get(field):
            raise ValueError(f"recomputed pair mismatch at {field}: {pair_path}")

    source = _mapping(sync.get("source"), "sync.source")
    if source.get("tree_clean") is not True:
        raise ValueError(f"source tree was not clean: {pair_path}")
    evidence = _mapping(sync.get("evidence"), "sync.evidence")
    if evidence.get("publishable_benchmark_evidence") is not True:
        raise ValueError(f"benchmark evidence is not publishable: {pair_path}")
    workload_data = _mapping(sync.get("workload"), "sync.workload")
    budget = _mapping(
        _mapping(sync.get("runtime"), "sync.runtime").get("budget"),
        "sync.runtime.budget",
    )
    if workload_data.get("id") != workload:
        raise ValueError(f"workload identity mismatch: {pair_path}")
    if workload_data.get("max_new_tokens") != tokens:
        raise ValueError(f"token-count mismatch: {pair_path}")
    if budget.get("slots_per_layer") != slots:
        raise ValueError(f"slot-count mismatch: {pair_path}")

    observations: list[dict[str, Any]] = []
    for pass_name, pass_label in PASS_LABELS.items():
        comparison = _mapping(
            _mapping(pair.get("passes"), "pair.passes").get(pass_name),
            f"pair.passes.{pass_name}",
        )
        metrics = dict(_mapping(comparison.get("metrics"), "pair metrics"))
        requests = _integer(metrics.get("requests"), "metrics.requests")
        hits = _integer(metrics.get("hits"), "metrics.hits")
        misses = _integer(metrics.get("misses"), "metrics.misses")
        if requests != hits + misses:
            raise ValueError(f"request accounting mismatch: {pair_path}")
        for field in (
            "admission_rejections",
            "coalesced_requests",
            "storage_failures",
            "transfer_failures",
        ):
            if _integer(metrics.get(field), f"metrics.{field}") != 0:
                raise ValueError(f"nonzero {field}: {pair_path}")

        sync_wall = _number(comparison.get("sync_wall_seconds"), "sync wall")
        async_wall = _number(comparison.get("async_wall_seconds"), "async wall")
        ratio = _number(comparison.get("sync_over_async_ratio"), "paired ratio")
        saving_fraction = _number(comparison.get("saving_fraction"), "saving fraction")
        if not math.isclose(ratio, sync_wall / async_wall, rel_tol=1e-12):
            raise ValueError(f"paired ratio mismatch: {pair_path}")
        if not math.isclose(
            saving_fraction, (sync_wall - async_wall) / sync_wall, rel_tol=1e-12
        ):
            raise ValueError(f"paired saving mismatch: {pair_path}")

        observations.append(
            {
                "group": group,
                "workload": workload,
                "tokens": tokens,
                "slots_per_layer": slots,
                "repetition": repetition,
                "order": PAIR_ORDERS[repetition],
                "cache_condition": pass_label,
                "sync_wall_seconds": sync_wall,
                "async_wall_seconds": async_wall,
                "sync_over_async_ratio": ratio,
                "saving_seconds": _number(
                    comparison.get("saving_seconds"), "saving seconds"
                ),
                "saving_fraction": saving_fraction,
                "sync_peak_allocated_vram_bytes": _integer(
                    comparison.get("sync_peak_allocated_vram_bytes"), "sync peak VRAM"
                ),
                "async_peak_allocated_vram_bytes": _integer(
                    comparison.get("async_peak_allocated_vram_bytes"), "async peak VRAM"
                ),
                "metrics": metrics,
            }
        )

    relative_dir = pair_path.parent.relative_to(study_root).as_posix()
    raw_hashes = {
        f"{relative_dir}/{name}": _sha256(pair_path.with_name(name))
        for name in ("sync.json", "async.json", "pair.json")
    }
    identity = {
        "source_commit": recomputed["source_commit"],
        "benchmark_script_sha256": recomputed["benchmark_script_sha256"],
        "model": _safe_model(sync),
        "environment": _safe_environment(sync),
        "created_at": sync.get("created_at"),
        "workload_file_sha256": workload_data.get("workload_file_sha256"),
        "policy": _mapping(sync.get("runtime"), "runtime").get("policy"),
        "expert_bytes": budget.get("expert_bytes"),
        "staging_slots": budget.get("staging_slots"),
    }
    return observations, raw_hashes, identity


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [_number(row["sync_over_async_ratio"], "ratio") for row in rows]
    savings = [_number(row["saving_fraction"], "saving") for row in rows]
    sync_times = [_number(row["sync_wall_seconds"], "sync wall") for row in rows]
    async_times = [_number(row["async_wall_seconds"], "async wall") for row in rows]
    ratio_median = _median(ratios)
    return {
        "repetitions": len(rows),
        "median_sync_wall_seconds": _median(sync_times),
        "median_async_wall_seconds": _median(async_times),
        "paired_ratio_median": ratio_median,
        "paired_ratio_min": min(ratios),
        "paired_ratio_max": max(ratios),
        "paired_ratio_mad": _mad(ratios),
        "paired_ratio_mad_over_median": _mad(ratios) / ratio_median,
        "paired_time_saved_fraction_median": _median(savings),
        "faster_repetitions": sum(ratio > 1.0 for ratio in ratios),
        "all_repetitions_faster": all(ratio > 1.0 for ratio in ratios),
    }


def _condition_aggregates(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        key = (
            row["group"],
            row["workload"],
            row["tokens"],
            row["slots_per_layer"],
            row["cache_condition"],
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group, workload, tokens, slots, condition = key
        rows = sorted(grouped[key], key=lambda item: item["repetition"])
        if [row["repetition"] for row in rows] != [1, 2, 3]:
            raise ValueError(f"incomplete repetitions for {key}")
        result.append(
            {
                "group": group,
                "workload": workload,
                "tokens": tokens,
                "slots_per_layer": slots,
                "cache_condition": condition,
                **_aggregate_rows(rows),
            }
        )
    return result


def _core_aggregate(observations: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in ("cold", "retained"):
        repetitions: list[dict[str, Any]] = []
        for repetition in range(1, 4):
            rows = [
                row
                for row in observations
                if row["group"] == "core"
                and row["cache_condition"] == condition
                and row["repetition"] == repetition
            ]
            if len(rows) != len(CORE_WORKLOADS):
                raise ValueError(
                    f"incomplete core aggregate: {condition} R{repetition}"
                )
            sync_wall = sum(row["sync_wall_seconds"] for row in rows)
            async_wall = sum(row["async_wall_seconds"] for row in rows)
            repetitions.append(
                {
                    "repetition": repetition,
                    "order": PAIR_ORDERS[repetition],
                    "sync_wall_seconds": sync_wall,
                    "async_wall_seconds": async_wall,
                    "sync_over_async_ratio": sync_wall / async_wall,
                    "saving_fraction": (sync_wall - async_wall) / sync_wall,
                }
            )
        result[condition] = {
            "workloads_per_repetition": len(CORE_WORKLOADS),
            "repetitions": repetitions,
            "aggregate": _aggregate_rows(repetitions),
        }
    return result


def build_summary(
    study_root: Path,
    *,
    raw_archive: Path | None = None,
    study_script: Path | None = None,
) -> dict[str, Any]:
    study_root = study_root.resolve()
    pair_paths = sorted(study_root.rglob("pair.json"))
    cases = {_case_from_path(study_root, path) for path in pair_paths}
    if len(cases) != len(pair_paths):
        raise ValueError("duplicate study cases")
    missing = EXPECTED_CASES - cases
    unexpected = cases - EXPECTED_CASES
    if missing or unexpected:
        raise ValueError(
            f"study matrix mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    comparator = _load_comparator()
    observations: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    identities: list[dict[str, Any]] = []
    for pair_path in pair_paths:
        case = _case_from_path(study_root, pair_path)
        rows, hashes, identity = _validate_case(study_root, pair_path, case, comparator)
        observations.extend(rows)
        raw_hashes.update(hashes)
        identities.append(identity)

    identity = identities[0]
    for candidate in identities[1:]:
        for field in (
            "source_commit",
            "benchmark_script_sha256",
            "model",
            "environment",
            "workload_file_sha256",
            "policy",
            "expert_bytes",
            "staging_slots",
        ):
            if candidate[field] != identity[field]:
                raise ValueError(f"cross-run identity mismatch at {field}")

    observations.sort(
        key=lambda row: (
            row["group"],
            row["workload"],
            row["tokens"],
            row["slots_per_layer"],
            row["repetition"],
            row["cache_condition"],
        )
    )
    aggregates = _condition_aggregates(observations)
    core = _core_aggregate(observations)
    cold_faster = sum(
        row["sync_over_async_ratio"] > 1.0
        for row in observations
        if row["cache_condition"] == "cold"
    )
    retained_faster = sum(
        row["sync_over_async_ratio"] > 1.0
        for row in observations
        if row["cache_condition"] == "retained"
    )
    measured_dates = sorted(
        {
            str(item["created_at"])[:10]
            for item in identities
            if isinstance(item.get("created_at"), str)
        }
    )

    source = {
        "commit": identity["source_commit"],
        "tree_clean": True,
        "benchmark_script": "scripts/benchmark_paged_olmoe.py",
        "benchmark_script_sha256": identity["benchmark_script_sha256"],
        "pair_comparator": "scripts/compare_paged_pipeline_pair.py",
        "pair_comparator_sha256": _sha256(COMPARATOR_PATH),
        "summarizer": "scripts/summarize_paged_runpod_study.py",
    }
    if study_script is not None:
        source["runpod_study_script_sha256"] = _sha256(study_script)
    if raw_archive is not None:
        source["raw_archive_sha256"] = _sha256(raw_archive)

    return {
        "schema_version": 1,
        "evidence_label": "36-pair RTX 6000 Ada sync-versus-async paged-runtime study",
        "measured_dates_utc": measured_dates,
        "source": source,
        "model": identity["model"],
        "environment": identity["environment"],
        "protocol": {
            "paired_cases": len(pair_paths),
            "benchmark_processes": len(pair_paths) * 2,
            "pass_comparisons": len(observations),
            "repetitions_per_case": 3,
            "alternating_order": [PAIR_ORDERS[index] for index in range(1, 4)],
            "policy": identity["policy"],
            "staging_slots": identity["staging_slots"],
            "expert_bytes": identity["expert_bytes"],
            "workload_file_sha256": identity["workload_file_sha256"],
            "os_page_cache": "explicitly warmed before each benchmark process",
            "decoding": "teacher-forced pinned greedy reference",
            "concurrency": 1,
            "pair_gate": "recomputed exact token, source, cache, traffic, memory-budget and failure-counter equality",
        },
        "matrix": {
            "core": {
                "workloads": list(CORE_WORKLOADS),
                "tokens": 16,
                "slots_per_layer": 32,
            },
            "length": {
                "workload": "python_code",
                "tokens": [2, 8, 32, 64],
                "slots_per_layer": 32,
            },
            "capacity": {
                "workload": "python_code",
                "tokens": 16,
                "slots_per_layer": [16, 24, 40],
            },
        },
        "correctness": {
            "all_pair_gates_passed": True,
            "pair_gates_passed": len(pair_paths),
            "all_exact_invariants": True,
            "all_admission_rejections_zero": True,
            "all_storage_failures_zero": True,
            "all_transfer_failures_zero": True,
            "all_coalesced_requests_zero": True,
        },
        "headline": {
            "cold_faster_comparisons": cold_faster,
            "cold_total_comparisons": len(pair_paths),
            "retained_faster_comparisons": retained_faster,
            "retained_total_comparisons": len(pair_paths),
            "core_cold_paired_ratio_median": core["cold"]["aggregate"][
                "paired_ratio_median"
            ],
            "core_cold_time_saved_fraction_median": core["cold"]["aggregate"][
                "paired_time_saved_fraction_median"
            ],
            "core_retained_paired_ratio_median": core["retained"]["aggregate"][
                "paired_ratio_median"
            ],
            "core_retained_time_saved_fraction_median": core["retained"]["aggregate"][
                "paired_time_saved_fraction_median"
            ],
        },
        "core_aggregate": core,
        "condition_aggregates": aggregates,
        "observations": observations,
        "raw_artifacts_sha256": dict(sorted(raw_hashes.items())),
        "limitations": [
            "This is one GPU, one pinned OLMoE checkpoint and one seed; it is not a universal serving claim.",
            "Teacher forcing fixes routing comparability but does not establish long free-running generation quality.",
            "The host page cache was intentionally warm; logical storage bytes are not physical NVMe telemetry.",
            "No concurrent requests, batching study or production server baseline was included.",
            "The study compares sync and async MoEVM scheduling, not a fully resident GPU runtime or a tuned serving engine.",
            "Three repetitions expose direction and spread but do not support a confidence interval or significance claim.",
            "CUDA event or profiler interval evidence is still required to attribute wall-time changes to true H2D/compute overlap.",
        ],
    }


def _encoded(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path)
    parser.add_argument("--study-script", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_summary(
        args.study_root,
        raw_archive=args.raw_archive,
        study_script=args.study_script,
    )
    encoded = _encoded(payload)
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != encoded
        ):
            raise SystemExit(f"summary is stale: {args.output}")
        print(f"Verified {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
