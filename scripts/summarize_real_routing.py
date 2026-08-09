#!/usr/bin/env python3
"""Build a portable, reviewable reference study from real-routing results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import tomllib
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(_REPO_ROOT / "results" / "real-routing" / "olmoe-0924"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "benchmarks" / "reference" / "real-routing-olmoe-m1"),
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "olmoe_1b_7b_0924.toml"),
    )
    parser.add_argument(
        "--workloads",
        default=str(_REPO_ROOT / "benchmarks" / "workloads" / "olmoe_m1.json"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "router_temporal_overlap": [
            record["router"]["mean_temporal_overlap"] for record in records
        ],
        "router_normalized_entropy": [
            record["router"]["mean_normalized_entropy"] for record in records
        ],
        "router_selected_probability_mass": [
            record["router"]["mean_topk_probability_mass"] for record in records
        ],
        "predictor_precision": [
            record["router"]["predictor"]["precision"] for record in records
        ],
        "predictor_recall": [
            record["router"]["predictor"]["recall"] for record in records
        ],
        "replay_speedup": [record["replay"]["speedup"] for record in records],
        "replay_demand_stall_reduction": [
            record["replay"]["demand_stall_reduction"] for record in records
        ],
        "replay_ram_to_vram_traffic_change": [
            record["replay"]["ram_to_vram_traffic_change"] for record in records
        ],
        "replay_prefetch_precision": [
            record["replay"]["prefetch_precision"] for record in records
        ],
    }
    return {
        "records": len(records),
        "tokens": sum(record["tokens"] for record in records),
        "routing_steps": sum(record["routing_steps"] for record in records),
        "expert_accesses": sum(record["expert_accesses"] for record in records),
        "metrics": {
            name: {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in metrics.items()
        },
    }


def _percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _markdown(study: dict[str, Any]) -> str:
    aggregate = study["aggregate"]
    metrics = aggregate["metrics"]
    rows = "\n".join(
        "| {seed} | {workload} | {tokens} | {overlap} | {precision} | "
        "{speedup:.4f}x | {traffic} |".format(
            seed=record["seed"],
            workload=record["workload_id"],
            tokens=record["tokens"],
            overlap=_percent(record["router"]["mean_temporal_overlap"]),
            precision=_percent(record["router"]["predictor"]["precision"]),
            speedup=record["replay"]["speedup"],
            traffic=_percent(record["replay"]["ram_to_vram_traffic_change"]),
        )
        for record in study["records"]
    )
    return f"""# OLMoE real-routing M1 reference study

> **Evidence boundary:** router decisions and scores were captured from the real
> pinned model. Cache speedups, stalls, and traffic are trace replays under the
> provisional simulator configuration; they are not measured runtime speedups.

## Scope

- Model: `{study["model"]["id"]}` at `{study["model"]["revision"]}`
- License: {study["model"]["license"]} (weights are not redistributed here)
- Workloads: {aggregate["records"]} captures across seeds 17 and 29
- Routing evidence: {aggregate["tokens"]} tokens, {aggregate["routing_steps"]:,}
  token/layer steps, {aggregate["expert_accesses"]:,} expert accesses
- Hardware: {study["environment"]["gpu"]}, CUDA {study["environment"]["cuda_runtime"]}

## Aggregate findings

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Real temporal overlap | {_percent(metrics["router_temporal_overlap"]["mean"])} | {_percent(metrics["router_temporal_overlap"]["min"])} | {_percent(metrics["router_temporal_overlap"]["max"])} |
| Real normalized entropy | {_percent(metrics["router_normalized_entropy"]["mean"])} | {_percent(metrics["router_normalized_entropy"]["min"])} | {_percent(metrics["router_normalized_entropy"]["max"])} |
| Online predictor precision | {_percent(metrics["predictor_precision"]["mean"])} | {_percent(metrics["predictor_precision"]["min"])} | {_percent(metrics["predictor_precision"]["max"])} |
| Simulated replay speedup | {metrics["replay_speedup"]["mean"]:.4f}x | {metrics["replay_speedup"]["min"]:.4f}x | {metrics["replay_speedup"]["max"]:.4f}x |
| Simulated demand-stall reduction | {_percent(metrics["replay_demand_stall_reduction"]["mean"])} | {_percent(metrics["replay_demand_stall_reduction"]["min"])} | {_percent(metrics["replay_demand_stall_reduction"]["max"])} |
| Simulated RAM-to-VRAM traffic change | {_percent(metrics["replay_ram_to_vram_traffic_change"]["mean"])} | {_percent(metrics["replay_ram_to_vram_traffic_change"]["min"])} | {_percent(metrics["replay_ram_to_vram_traffic_change"]["max"])} |

The current predictor is only approximately neutral on these real traces: its
mean replay speedup is `{metrics["replay_speedup"]["mean"]:.4f}x`, while it changes
RAM-to-VRAM traffic by `{_percent(metrics["replay_ram_to_vram_traffic_change"]["mean"])}`.
This is useful negative evidence: the strong synthetic result does not transfer
unchanged to this model and workload set.

## Per capture

| Seed | Workload | Tokens | Real overlap | Predictor precision | Replay speedup | Replay traffic change |
|---:|---|---:|---:|---:|---:|---:|
{rows}

## Reproduction

Every committed JSONL trace includes router scores and has its SHA-256 recorded
in `study.json`. Replay it with:

```bash
python -m moevm compare --config configs/olmoe_1b_7b_0924.toml \\
  --trace benchmarks/reference/real-routing-olmoe-m1/traces/seed-17/systems_en.trace.jsonl \\
  --no-write
```
"""


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    workload_path = Path(args.workloads).resolve()
    workload_payload = _read_json(workload_path)
    workload_lookup = {item["id"]: item for item in workload_payload["workloads"]}

    records: list[dict[str, Any]] = []
    for manifest_path in sorted(input_dir.glob("seed-*/manifest.json")):
        manifest = _read_json(manifest_path)
        seed = int(manifest["seed"])
        seed_dir = manifest_path.parent
        for source_record in manifest["records"]:
            workload_id = source_record["workload_id"]
            metadata = _read_json(seed_dir / f"{workload_id}.metadata.json")
            analysis = _read_json(
                seed_dir / f"{workload_id}.analysis" / "routing_analysis.json"
            )["analysis"]
            comparison = _read_json(
                seed_dir / f"{workload_id}.replay" / "comparison.json"
            )
            trace_source = seed_dir / f"{workload_id}.trace.jsonl"
            trace_hash = _sha256(trace_source)
            if trace_hash != metadata["trace"]["sha256"]:
                raise ValueError(f"trace hash mismatch: {trace_source}")

            trace_relative = (
                Path("traces") / f"seed-{seed}" / f"{workload_id}.trace.jsonl"
            )
            trace_target = output_dir / trace_relative
            trace_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trace_source, trace_target)

            analysis_relative = (
                Path("analysis") / f"seed-{seed}" / f"{workload_id}.json"
            )
            _write_json(
                output_dir / analysis_relative,
                {
                    "label": "routing capture",
                    "trace": {
                        "path": trace_relative.as_posix(),
                        "sha256": trace_hash,
                    },
                    "analysis": analysis,
                },
            )
            replay_relative = Path("replays") / f"seed-{seed}" / f"{workload_id}.json"
            _write_json(output_dir / replay_relative, comparison)

            predictor = analysis["predictor"]
            replay = comparison["comparison"]
            records.append(
                {
                    "seed": seed,
                    "temperature": manifest["temperature"],
                    "workload_id": workload_id,
                    "category": workload_lookup[workload_id]["category"],
                    "language": workload_lookup[workload_id]["language"],
                    "tokens": analysis["tokens"],
                    "routing_steps": analysis["steps"],
                    "expert_accesses": analysis["expert_accesses"],
                    "trace": {
                        "path": trace_relative.as_posix(),
                        "sha256": trace_hash,
                    },
                    "analysis_path": analysis_relative.as_posix(),
                    "replay_path": replay_relative.as_posix(),
                    "router": {
                        "score_coverage": analysis["score_coverage"],
                        "mean_temporal_overlap": analysis["mean_temporal_overlap"],
                        "exact_repeat_rate": analysis["exact_repeat_rate"],
                        "mean_normalized_entropy": analysis["mean_normalized_entropy"],
                        "mean_topk_probability_mass": analysis[
                            "mean_topk_probability_mass"
                        ],
                        "predictor": predictor,
                    },
                    "replay": {
                        "evidence_label": "trace replay; simulated timing",
                        "speedup": replay["speedup"],
                        "demand_stall_reduction": replay["demand_stall_reduction"],
                        "demand_nvme_reduction": replay["demand_nvme_reduction"],
                        "total_nvme_reduction": replay["total_nvme_reduction"],
                        "ram_to_vram_traffic_change": replay[
                            "ram_to_vram_traffic_change"
                        ],
                        "baseline_l1_hit_rate": comparison["baseline"][
                            "demand_l1_hit_rate"
                        ],
                        "prefetch_l1_hit_rate": comparison["prefetch"][
                            "demand_l1_hit_rate"
                        ],
                        "prefetch_precision": comparison["prefetch"][
                            "prefetch_precision"
                        ],
                    },
                    "capture_observation": {
                        "prefill_seconds": metadata["timing_observation"][
                            "prefill_seconds"
                        ],
                        "decode_seconds": metadata["timing_observation"][
                            "decode_seconds"
                        ],
                        "peak_vram_bytes": metadata["environment"]["peak_vram_bytes"],
                        "warning": metadata["timing_observation"]["warning"],
                    },
                }
            )

    if not records:
        raise ValueError(f"no capture manifests found under {input_dir}")
    records.sort(key=lambda record: (record["seed"], record["workload_id"]))
    first_metadata = _read_json(
        input_dir
        / f"seed-{records[0]['seed']}"
        / f"{records[0]['workload_id']}.metadata.json"
    )
    model = first_metadata["model"]
    project_metadata = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    study = {
        "schema_version": 1,
        "project": {
            "name": project_metadata["name"],
            "version": project_metadata["version"],
            "source_ref": f"v{project_metadata['version']}",
        },
        "evidence_boundary": {
            "router": "captured from the real pinned model",
            "replay": "simulated using provisional transfer and compute parameters",
            "capture_timing": "Accelerate CPU-offload observation; not an MoEVM runtime benchmark",
        },
        "model": {
            "id": model["id"],
            "revision": model["resolved_revision"],
            "license": model["license"],
            "layers": model["layers"],
            "experts_per_layer": model["experts_per_layer"],
            "top_k": model["top_k"],
            "checkpoint_shards_sha256": {
                "model-00001-of-00003.safetensors": "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e",
                "model-00002-of-00003.safetensors": "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7",
                "model-00003-of-00003.safetensors": "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04",
            },
        },
        "environment": {
            "gpu": first_metadata["environment"]["gpu"],
            "cuda_runtime": first_metadata["environment"]["cuda_runtime"],
            "python": first_metadata["environment"]["python"],
            "packages": first_metadata["environment"]["packages"],
        },
        "inputs": {
            "config": "configs/olmoe_1b_7b_0924.toml",
            "config_sha256": _sha256(config_path),
            "workloads": "benchmarks/workloads/olmoe_m1.json",
            "workloads_sha256": _sha256(workload_path),
        },
        "aggregate": _aggregate(records),
        "by_seed": {
            str(seed): _aggregate(
                [record for record in records if record["seed"] == seed]
            )
            for seed in sorted({record["seed"] for record in records})
        },
        "records": records,
    }
    _write_json(output_dir / "study.json", study)
    (output_dir / "README.md").write_text(_markdown(study), encoding="utf-8")
    print(
        f"Wrote {len(records)} portable records, {study['aggregate']['tokens']} "
        f"tokens, and {study['aggregate']['expert_accesses']} expert accesses "
        f"to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
