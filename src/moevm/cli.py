from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ExperimentConfig, load_config
from .report import comparison_console, write_comparison
from .simulator import compare_experiment, run_experiment
from .trace import SyntheticRoutingTrace, read_trace, write_trace


def _load_steps(config: ExperimentConfig, trace_path: str | None):
    if trace_path:
        return read_trace(trace_path)
    return list(SyntheticRoutingTrace(config).generate())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moevm",
        description="Research simulator for expert-aware MoE memory virtualization.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="compare LRU baseline and predictive prefetch")
    compare.add_argument("--config", default="configs/toy.toml", help="TOML experiment config")
    compare.add_argument("--trace", help="optional JSONL routing trace")
    compare.add_argument("--tokens", type=int, help="override synthetic token count")
    compare.add_argument("--output-dir", default="results/latest", help="report directory")
    compare.add_argument("--no-write", action="store_true", help="print without writing reports")

    run = subparsers.add_parser("run", help="run one simulation mode")
    run.add_argument("--config", default="configs/toy.toml")
    run.add_argument("--trace", help="optional JSONL routing trace")
    run.add_argument("--tokens", type=int, help="override synthetic token count")
    run.add_argument("--mode", choices=("baseline", "prefetch"), default="prefetch")
    run.add_argument("--json", action="store_true", help="print JSON metrics")

    trace = subparsers.add_parser("trace", help="generate a synthetic routing trace")
    trace.add_argument("--config", default="configs/toy.toml")
    trace.add_argument("--tokens", type=int, help="override token count")
    trace.add_argument("--output", default="results/synthetic.trace.jsonl")

    doctor = subparsers.add_parser("doctor", help="validate a config and show cache capacity")
    doctor.add_argument("--config", default="configs/toy.toml")

    return parser


def _doctor(config: ExperimentConfig) -> str:
    expert_bytes = config.model.expert_size_bytes
    vram_slots = config.hardware.vram_cache_bytes // expert_bytes
    ram_slots = config.hardware.ram_cache_bytes // expert_bytes
    return "\n".join(
        [
            "Configuration is valid.",
            f"Model: {config.model.name}",
            f"Expert bytes (synthetic): {expert_bytes:,}",
            f"VRAM cache slots: {vram_slots:,}",
            f"RAM cache slots: {ram_slots:,}",
            f"Steps: {config.trace.tokens * config.model.layers:,}",
            f"Expert accesses: {config.trace.tokens * config.model.layers * config.model.top_k:,}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if hasattr(args, "tokens"):
            config = config.with_tokens(args.tokens)

        if args.command == "compare":
            steps = _load_steps(config, args.trace)
            result = compare_experiment(config, trace=steps)
            print(comparison_console(result, config))
            if not args.no_write:
                json_path, markdown_path = write_comparison(args.output_dir, result, config)
                print(f"\nWrote {json_path} and {markdown_path}")
            return 0

        if args.command == "run":
            metrics = run_experiment(config, mode=args.mode, trace=_load_steps(config, args.trace))
            if args.json:
                print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
            else:
                print(
                    f"{metrics.mode}: {metrics.tokens_per_second:.3f} estimated tok/s; "
                    f"L1 hit-rate {metrics.demand_l1_hit_rate * 100:.2f}%; "
                    f"NVMe bytes {metrics.total_nvme_to_ram_bytes:,}"
                )
            return 0

        if args.command == "trace":
            destination = Path(args.output)
            steps = list(SyntheticRoutingTrace(config).generate())
            write_trace(destination, steps)
            print(f"Wrote {len(steps):,} routing steps to {destination}")
            return 0

        if args.command == "doctor":
            print(_doctor(config))
            return 0

        parser.error(f"unknown command: {args.command}")
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
