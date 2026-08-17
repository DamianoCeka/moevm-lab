from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from . import __version__
from .analysis import (
    analyze_routing_trace,
    trace_analysis_console,
    write_trace_analysis,
)
from .config import ExperimentConfig, load_config
from .machine_doctor import MachineReport, collect_machine_report
from .placement_analysis import (
    PLACEMENT_POLICIES,
    PlacementTrace,
    analyze_placement_leave_one_workload_out,
    analyze_placement_train_test,
    discover_trace_paths,
    load_placement_traces,
    placement_analysis_console,
    write_placement_analysis,
)
from .report import comparison_console, write_comparison
from .simulator import compare_experiment, run_experiment
from .trace import SyntheticRoutingTrace, read_trace, write_trace


def _configure_windows_stdio() -> None:
    """Keep Unicode reports printable in legacy Windows shells and pipes."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Embedded callers may provide a closed or non-reconfigurable stream.
            pass


def _bundled_config(name: str) -> str:
    return str(Path(__file__).resolve().with_name("configs") / name)


def _load_steps(config: ExperimentConfig, trace_path: str | None):
    if trace_path:
        return read_trace(trace_path)
    return list(SyntheticRoutingTrace(config).generate())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moevm",
        description="Research simulator for expert-aware MoE memory virtualization.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser(
        "compare", help="compare LRU baseline and predictive prefetch"
    )
    compare.add_argument(
        "--config",
        default=_bundled_config("toy.toml"),
        help="TOML experiment config",
    )
    compare.add_argument("--trace", help="optional JSONL routing trace")
    compare.add_argument("--tokens", type=int, help="override synthetic token count")
    compare.add_argument(
        "--output-dir", default="results/latest", help="report directory"
    )
    compare.add_argument(
        "--no-write", action="store_true", help="print without writing reports"
    )

    run = subparsers.add_parser("run", help="run one simulation mode")
    run.add_argument("--config", default=_bundled_config("toy.toml"))
    run.add_argument("--trace", help="optional JSONL routing trace")
    run.add_argument("--tokens", type=int, help="override synthetic token count")
    run.add_argument("--mode", choices=("baseline", "prefetch"), default="prefetch")
    run.add_argument("--json", action="store_true", help="print JSON metrics")

    trace = subparsers.add_parser("trace", help="generate a synthetic routing trace")
    trace.add_argument("--config", default=_bundled_config("toy.toml"))
    trace.add_argument("--tokens", type=int, help="override token count")
    trace.add_argument("--output", default="results/synthetic.trace.jsonl")

    doctor = subparsers.add_parser(
        "doctor", help="validate a config and inspect its memory budget"
    )
    doctor.add_argument("--config", default=_bundled_config("toy.toml"))
    doctor.add_argument(
        "--machine",
        action="store_true",
        help="add read-only GPU, RAM, and disk observations",
    )
    doctor.add_argument(
        "--cache-path",
        default=".",
        help="path whose existing volume should be inspected with --machine",
    )
    doctor.add_argument(
        "--no-gpu-probe",
        action="store_true",
        help="skip the optional nvidia-smi call with --machine",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="emit the read-only --machine report as JSON",
    )

    analyze = subparsers.add_parser(
        "analyze-trace",
        help="measure locality, router confidence, and online predictability",
    )
    analyze.add_argument("--config", default=_bundled_config("toy.toml"))
    analyze.add_argument("--trace", required=True, help="JSONL routing trace")
    analyze.add_argument(
        "--output-dir",
        default="results/trace-analysis",
        help="analysis report directory",
    )
    analyze.add_argument(
        "--no-write", action="store_true", help="print without writing reports"
    )

    placement = subparsers.add_parser(
        "analyze-placement",
        help="evaluate auditable cache placement policies on JSONL traces",
    )
    placement.add_argument("--config", required=True, help="model configuration")
    placement.add_argument(
        "--train-trace", action="append", default=[], help="training JSONL trace"
    )
    placement.add_argument(
        "--train-dir",
        action="append",
        default=[],
        help="directory of training *.trace.jsonl files",
    )
    placement.add_argument(
        "--test-trace", action="append", default=[], help="test JSONL trace"
    )
    placement.add_argument(
        "--test-dir",
        action="append",
        default=[],
        help="directory of test *.trace.jsonl files",
    )
    placement.add_argument(
        "--protocol",
        choices=("train-test", "leave-one-workload-out"),
        default="train-test",
    )
    placement.add_argument("--capacity-per-layer", type=int, required=True)
    placement.add_argument(
        "--layer-capacity",
        action="append",
        default=[],
        metavar="LAYER=SLOTS",
        help="override one layer; when used, specify every differing layer",
    )
    placement.add_argument("--protected-hot", type=int, default=0)
    placement.add_argument(
        "--policy",
        action="append",
        choices=PLACEMENT_POLICIES,
        help="policy to evaluate; repeat to select multiple (default: all)",
    )
    placement.add_argument(
        "--output",
        default="results/placement-analysis.json",
        help="JSON report path",
    )

    return parser


def _placement_capacities(
    default: int, overrides: list[str], layer_count: int
) -> int | dict[int, int]:
    if not overrides:
        return default
    parsed = {layer: default for layer in range(layer_count)}
    overridden: set[int] = set()
    for raw_override in overrides:
        try:
            raw_layer, raw_slots = raw_override.split("=", maxsplit=1)
            layer = int(raw_layer)
            slots = int(raw_slots)
        except ValueError as exc:
            raise ValueError(
                f"invalid --layer-capacity {raw_override!r}; expected LAYER=SLOTS"
            ) from exc
        if layer < 0 or slots < 0:
            raise ValueError("layer-capacity values cannot be negative")
        if layer >= layer_count:
            raise ValueError(f"layer-capacity layer exceeds model: {layer}")
        if layer in overridden:
            raise ValueError(f"duplicate layer-capacity override: {layer}")
        parsed[layer] = slots
        overridden.add(layer)
    return parsed


def _validate_placement_shape(
    config: ExperimentConfig, traces: tuple[PlacementTrace, ...]
) -> None:
    expected_layers = tuple(range(config.model.layers))
    for trace in traces:
        layers = tuple(sorted({step.layer_index for step in trace.steps}))
        if layers != expected_layers:
            raise ValueError(
                "placement trace layer count does not match model.layers: "
                f"{len(layers)} != {config.model.layers} ({trace.source})"
            )
        if any(len(step.experts) != config.model.top_k for step in trace.steps):
            raise ValueError(
                f"placement trace top-k does not match model.top_k: {trace.source}"
            )
        if any(
            expert >= config.model.experts_per_layer
            for step in trace.steps
            for expert in step.experts
        ):
            raise ValueError(
                "placement trace expert id exceeds model.experts_per_layer: "
                f"{trace.source}"
            )


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


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if value < 1024**2:
        return f"{value:,} bytes"
    if value < 1024**3:
        return f"{value / (1024**2):.2f} MiB"
    return f"{value / (1024**3):.2f} GiB"


def _machine_doctor_console(report: MachineReport) -> str:
    """Render a human-readable snapshot without implying runtime readiness."""
    ledger = report.ledger
    memory = report.system_memory
    disk = report.disk
    lines = [
        "Machine observations (read-only; no CUDA context, model load, or download).",
        (
            f"Host RAM [{memory.status}, {memory.source}]: "
            f"{_format_bytes(memory.available_bytes)} available / "
            f"{_format_bytes(memory.total_bytes)} total"
        ),
        (
            f"Disk [{disk.status}] for {disk.requested_path}: "
            f"{_format_bytes(disk.free_bytes)} free / "
            f"{_format_bytes(disk.total_bytes)} total"
        ),
    ]
    if memory.detail:
        lines.append(f"Host RAM detail: {memory.detail}")
    if disk.detail:
        lines.append(f"Disk detail: {disk.detail}")
    if disk.inspected_path and disk.inspected_path != disk.requested_path:
        lines.append(f"Disk inspected ancestor: {disk.inspected_path}")

    if report.gpu is None:
        lines.append("GPU probe: skipped")
    elif report.gpu.status != "available":
        detail = f" ({report.gpu.detail})" if report.gpu.detail else ""
        lines.append(f"GPU probe [{report.gpu.status}]{detail}")
    else:
        lines.append("GPU probe [available]:")
        for gpu in report.gpu.gpus:
            lines.append(
                f"  GPU {gpu.index}: {gpu.name}; "
                f"{_format_bytes(gpu.free_vram_bytes)} free / "
                f"{_format_bytes(gpu.total_vram_bytes)} total; "
                f"driver {gpu.driver_version}"
            )

    lines.extend(
        [
            "",
            "Configuration memory ledger (logical config values, not checkpoint or physical-I/O measurements):",
            f"Model shape: {ledger.layers} layers × {ledger.experts_per_layer} experts; top-{ledger.top_k}",
            f"Average expert payload: {_format_bytes(ledger.expert_bytes)}",
            f"All expert payloads (logical): {_format_bytes(ledger.all_expert_logical_bytes)}",
            (
                f"VRAM expert budget: {_format_bytes(ledger.vram_cache_bytes)} "
                f"({ledger.vram_cache_slots:,} whole-expert slots; "
                f"{_format_bytes(ledger.vram_cache_remainder_bytes)} unused)"
            ),
            (
                f"RAM expert budget: {_format_bytes(ledger.ram_cache_bytes)} "
                f"({ledger.ram_cache_slots:,} whole-expert slots; "
                f"{_format_bytes(ledger.ram_cache_remainder_bytes)} unused)"
            ),
            (
                f"One token before cache hits: {ledger.routed_expert_accesses_per_token:,} "
                f"routed expert accesses; "
                f"{_format_bytes(ledger.per_token_logical_routed_expert_bytes)} "
                "logical payload"
            ),
            (
                "This is capacity and logical-flow planning, not a claim that the "
                "model fits, that the disk was read, or that a benchmark will be fast."
            ),
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _configure_windows_stdio()
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
                json_path, markdown_path = write_comparison(
                    args.output_dir, result, config
                )
                print(f"\nWrote {json_path} and {markdown_path}")
            return 0

        if args.command == "run":
            metrics = run_experiment(
                config, mode=args.mode, trace=_load_steps(config, args.trace)
            )
            if args.json:
                print(
                    json.dumps(
                        metrics.to_dict(),
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
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
            if args.json and not args.machine:
                raise ValueError("--json requires doctor --machine")
            if args.machine:
                report = collect_machine_report(
                    config,
                    disk_path=args.cache_path,
                    probe_gpu=not args.no_gpu_probe,
                )
                if args.json:
                    print(
                        json.dumps(
                            dataclasses.asdict(report),
                            allow_nan=False,
                            indent=2,
                            sort_keys=True,
                        )
                    )
                else:
                    print(_machine_doctor_console(report))
                return 0
            print(_doctor(config))
            return 0

        if args.command == "analyze-trace":
            steps = read_trace(args.trace)
            analysis = analyze_routing_trace(
                steps,
                experts_per_layer=config.model.experts_per_layer,
                predictor_config=config.predictor,
            )
            if analysis.layers != config.model.layers:
                raise ValueError(
                    "trace layer count does not match model.layers: "
                    f"{analysis.layers} != {config.model.layers}"
                )
            if analysis.top_k != config.model.top_k:
                raise ValueError(
                    "trace top-k does not match model.top_k: "
                    f"{analysis.top_k} != {config.model.top_k}"
                )
            print(trace_analysis_console(analysis))
            if not args.no_write:
                json_path, markdown_path = write_trace_analysis(
                    args.output_dir,
                    analysis,
                    trace_path=args.trace,
                )
                print(f"\nWrote {json_path} and {markdown_path}")
            return 0

        if args.command == "analyze-placement":
            train_paths = discover_trace_paths(
                traces=args.train_trace, directories=args.train_dir
            )
            test_paths = discover_trace_paths(
                traces=args.test_trace, directories=args.test_dir
            )
            train = load_placement_traces(train_paths)
            test = load_placement_traces(test_paths)
            _validate_placement_shape(config, train + test)
            options = {
                "capacity_per_layer": _placement_capacities(
                    args.capacity_per_layer,
                    args.layer_capacity,
                    config.model.layers,
                ),
                "protected_hot": args.protected_hot,
                "expert_bytes": config.model.expert_size_bytes,
                "policies": args.policy or PLACEMENT_POLICIES,
            }
            if args.protocol == "train-test":
                report = analyze_placement_train_test(train, test, **options)
            else:
                report = analyze_placement_leave_one_workload_out(
                    train, test, **options
                )
            print(placement_analysis_console(report))
            output_path = write_placement_analysis(args.output, report)
            print(f"\nWrote {output_path}")
            return 0

        parser.error(f"unknown command: {args.command}")
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
