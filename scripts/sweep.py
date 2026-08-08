#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moevm.config import load_config
from moevm.simulator import compare_experiment
from moevm.trace import SyntheticRoutingTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep VRAM expert-cache sizes.")
    parser.add_argument("--config", default="configs/toy.toml")
    parser.add_argument("--sizes-mib", default="64,96,128,192,256,384,512")
    parser.add_argument("--output", default="results/vram_sweep.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sizes = [
        float(value.strip()) for value in args.sizes_mib.split(",") if value.strip()
    ]
    if not sizes or min(sizes) <= 0:
        raise ValueError("--sizes-mib must contain positive numbers")

    rows: list[dict[str, object]] = []
    for size_mib in sizes:
        prefetch_bytes = config.model.expert_size_mib * config.predictor.prefetch_count
        fraction = min(0.49, prefetch_bytes / size_mib)
        experiment = replace(
            config,
            hardware=replace(
                config.hardware,
                vram_cache_mib=size_mib,
                prefetch_vram_fraction=fraction,
            ),
        )
        trace = list(SyntheticRoutingTrace(experiment).generate())
        result = compare_experiment(experiment, trace=trace)
        rows.append(
            {
                "vram_cache_mib": size_mib,
                "prefetch_fraction": fraction,
                "baseline_tok_s": result.baseline.tokens_per_second,
                "prefetch_tok_s": result.prefetch.tokens_per_second,
                "speedup": result.speedup,
                "baseline_l1_hit_rate": result.baseline.demand_l1_hit_rate,
                "prefetch_l1_hit_rate": result.prefetch.demand_l1_hit_rate,
                "prefetch_precision": result.prefetch.prefetch_precision,
                "demand_stall_reduction": result.demand_stall_reduction,
                "ram_to_vram_traffic_change": result.ram_to_vram_traffic_change,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
