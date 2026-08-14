"""Render the committed sync-versus-async OLMoE smoke as a deterministic SVG."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

WIDTH = 1200
HEIGHT = 900
CONDITION_ORDER = ("cold_expert_cache", "retained_expert_cache")
EXPECTED_ORDERS = ("async-sync", "sync-async", "async-sync")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _close(actual: object, expected: float, name: str) -> None:
    value = _number(actual, name)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} is {value}, expected {expected}")


def validate_reference(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("reference schema_version must be 1")
    source = _mapping(payload.get("source"), "source")
    if source.get("tree_clean") is not True:
        raise ValueError("benchmark source must be a clean tree")
    if not isinstance(source.get("commit"), str) or len(source["commit"]) != 40:
        raise ValueError("source.commit must be a full Git commit")
    for field in ("benchmark_script_sha256", "pair_comparator_sha256"):
        value = source.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"source.{field} must be a SHA-256 digest")

    protocol = _mapping(payload.get("protocol"), "protocol")
    if protocol.get("paired_repetitions") != 3:
        raise ValueError("the chart contract requires exactly three pairs")
    if tuple(protocol.get("alternating_order", ())) != EXPECTED_ORDERS:
        raise ValueError("unexpected alternating order")

    correctness = _mapping(payload.get("correctness"), "correctness")
    required_true = (
        "all_pair_gates_passed",
        "cross_repetition_identity_exact",
        "generated_and_fed_token_ids_exact",
        "cache_and_transfer_primitives_exact",
        "all_admission_rejections_zero",
        "all_storage_failures_zero",
        "all_transfer_failures_zero",
        "all_coalesced_requests_zero",
        "peak_allocated_vram_equal_between_modes",
    )
    for field in required_true:
        if correctness.get(field) is not True:
            raise ValueError(f"correctness.{field} must be true")

    conditions = _mapping(payload.get("conditions"), "conditions")
    for condition_name in CONDITION_ORDER:
        condition = _mapping(conditions.get(condition_name), condition_name)
        repetitions = condition.get("repetitions")
        if not isinstance(repetitions, list) or len(repetitions) != 3:
            raise ValueError(f"{condition_name} must contain three repetitions")
        ratios: list[float] = []
        savings: list[float] = []
        fractions: list[float] = []
        sync_times: list[float] = []
        async_times: list[float] = []
        for index, row_value in enumerate(repetitions, start=1):
            row = _mapping(row_value, f"{condition_name}.repetition[{index}]")
            if (
                row.get("repetition") != index
                or row.get("order") != EXPECTED_ORDERS[index - 1]
            ):
                raise ValueError(f"{condition_name} repetition identity mismatch")
            sync_time = _number(row.get("sync_wall_seconds"), "sync wall")
            async_time = _number(row.get("async_wall_seconds"), "async wall")
            if sync_time <= 0.0 or async_time <= 0.0 or async_time >= sync_time:
                raise ValueError(
                    "each validated pair must have positive async improvement"
                )
            ratio = sync_time / async_time
            saving = sync_time - async_time
            fraction = saving / sync_time
            _close(row.get("sync_over_async_ratio"), ratio, "paired ratio")
            _close(row.get("saving_seconds"), saving, "saving seconds")
            _close(row.get("saving_fraction"), fraction, "saving fraction")
            sync_times.append(sync_time)
            async_times.append(async_time)
            ratios.append(ratio)
            savings.append(saving)
            fractions.append(fraction)

        aggregate = _mapping(condition.get("aggregate"), f"{condition_name}.aggregate")
        ratio_median = statistics.median(ratios)
        ratio_mad = statistics.median(abs(value - ratio_median) for value in ratios)
        expected = {
            "median_sync_wall_seconds": statistics.median(sync_times),
            "median_async_wall_seconds": statistics.median(async_times),
            "paired_ratio_median": ratio_median,
            "paired_ratio_min": min(ratios),
            "paired_ratio_max": max(ratios),
            "paired_ratio_mad": ratio_mad,
            "paired_ratio_mad_over_median": ratio_mad / ratio_median,
            "paired_saving_seconds_median": statistics.median(savings),
            "paired_time_saved_fraction_median": statistics.median(fractions),
        }
        for field, value in expected.items():
            _close(aggregate.get(field), value, f"{condition_name}.{field}")
        if aggregate.get("all_repetitions_faster") is not True:
            raise ValueError(f"{condition_name} must mark all repetitions faster")


def _x(seconds: float) -> float:
    plot_left = 225.0
    plot_width = 700.0
    return plot_left + (seconds / 4.0) * plot_width


def _text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str = "body",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}" '
        f'text-anchor="{anchor}">{value}</text>'
    )


def render_svg(payload: dict[str, Any]) -> str:
    validate_reference(payload)
    conditions = _mapping(payload["conditions"], "conditions")
    commit = _mapping(payload["source"], "source")["commit"][:7]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">'
        ),
        '<title id="chart-title">Paged OLMoE runtime: synchronous versus asynchronous expert pipeline</title>',
        '<desc id="chart-desc">Six paired wall-time observations across empty and retained dynamic expert-cache conditions. In every pair the asynchronous MoEVM pipeline uses less wall time than the synchronous path, while validated tokens, cache counters, transfer bytes, and peak allocated VRAM remain identical.</desc>',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#111827}",
        ".title{font-size:30px;font-weight:700}",
        ".subtitle{font-size:15px;fill:#4B5563}",
        ".panel-title{font-size:18px;font-weight:700}",
        ".body{font-size:14px}",
        ".small{font-size:14px;fill:#4B5563}",
        ".value{font-size:14px;font-weight:650}",
        ".saving{font-size:14px;font-weight:700;fill:#004B76}",
        ".callout{font-size:14px;font-weight:650;fill:#004B76}",
        ".mono{font-family:Consolas,ui-monospace,monospace}",
        "</style>",
        '<rect width="1200" height="900" fill="#FFFFFF"/>',
        '<rect x="42" y="28" width="1116" height="830" rx="14" fill="#FFFFFF" stroke="#D1D5DB"/>',
        _text(68, 72, "Paged OLMoE runtime: sync vs async", css_class="title"),
        _text(
            68,
            101,
            "RTX 3080 Ti · LRU32/layer · 2 teacher-forced tokens · 3 paired repetitions",
            css_class="subtitle",
        ),
        '<rect x="924" y="52" width="194" height="40" rx="20" fill="#E6F4FA" stroke="#8CC6E3"/>',
        _text(1021, 78, "Lower wall time: 6 / 6", css_class="callout", anchor="middle"),
        '<rect x="69" y="126" width="12" height="12" fill="#5B6472"/>',
        _text(89, 137, "Sync path", css_class="small"),
        '<circle cx="306" cy="132" r="7" fill="#0072B2"/>',
        _text(321, 137, "MoEVM async", css_class="small"),
        _text(
            575,
            137,
            "Wall time (s, lower is better)",
            css_class="small",
            anchor="middle",
        ),
        _text(1062, 137, "Wall-time reduction", css_class="small", anchor="middle"),
        '<rect x="62" y="157" width="1076" height="252" rx="10" fill="#F9FAFB" stroke="#E5E7EB"/>',
        '<rect x="62" y="429" width="1076" height="252" rx="10" fill="#F9FAFB" stroke="#E5E7EB"/>',
    ]

    for tick in range(5):
        x = _x(float(tick))
        lines.append(
            f'<line x1="{x:.1f}" y1="199" x2="{x:.1f}" y2="633" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(_text(x, 704, str(tick), css_class="small mono", anchor="middle"))
    lines.append('<line x1="225" y1="685" x2="925" y2="685" stroke="#6B7280"/>')

    group_specs = (
        (
            "cold_expert_cache",
            "Empty dynamic expert cache",
            187.0,
            (246.0, 294.0, 342.0),
            385.0,
        ),
        (
            "retained_expert_cache",
            "Retained dynamic expert cache",
            459.0,
            (518.0, 566.0, 614.0),
            657.0,
        ),
    )
    for condition_name, title, title_y, row_y_values, callout_y in group_specs:
        condition = _mapping(conditions[condition_name], condition_name)
        lines.append(_text(82, title_y, title, css_class="panel-title"))
        for row, y in zip(condition["repetitions"], row_y_values, strict=True):
            sync_time = float(row["sync_wall_seconds"])
            async_time = float(row["async_wall_seconds"])
            sync_x = _x(sync_time)
            async_x = _x(async_time)
            lines.extend(
                [
                    _text(92, y + 5, f"R{row['repetition']}", css_class="body mono"),
                    f'<line x1="{async_x:.1f}" y1="{y:.1f}" x2="{sync_x:.1f}" y2="{y:.1f}" stroke="#7C8798" stroke-width="2"/>',
                    f'<circle cx="{async_x:.1f}" cy="{y:.1f}" r="7" fill="#0072B2" stroke="#004B76"/>',
                    f'<rect x="{sync_x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" fill="#5B6472" stroke="#374151"/>',
                    _text(
                        async_x - 11,
                        y + 4,
                        f"{async_time:.3f}s",
                        css_class="value mono",
                        anchor="end",
                    ),
                    _text(
                        sync_x + 11, y + 4, f"{sync_time:.3f}s", css_class="value mono"
                    ),
                    _text(
                        1062,
                        y + 5,
                        f"{float(row['saving_fraction']) * 100:.1f}%",
                        css_class="saving mono",
                        anchor="middle",
                    ),
                ]
            )
        aggregate = _mapping(condition["aggregate"], f"{condition_name}.aggregate")
        lines.append(
            f'<rect x="744" y="{callout_y - 25:.1f}" width="365" height="34" rx="17" fill="#E6F4FA"/>'
        )
        lines.append(
            _text(
                926,
                callout_y - 3,
                f"Median: {float(aggregate['paired_time_saved_fraction_median']) * 100:.1f}% less time · sync/async {float(aggregate['paired_ratio_median']):.2f}×",
                css_class="callout mono",
                anchor="middle",
            )
        )

    lines.extend(
        [
            '<rect x="62" y="735" width="1076" height="99" rx="10" fill="#F3F4F6"/>',
            _text(
                82,
                764,
                "Same validated work in both modes: token IDs, logical hit/miss/eviction counts, transfer bytes and peak VRAM.",
                css_class="body",
            ),
            _text(
                82,
                791,
                "n=3 pairs · one workload · two tokens. “Cold” means empty GPU expert cache, not cold SSD or OS cache.",
                css_class="small",
            ),
            _text(
                82,
                816,
                f"This smoke does not prove physical NVMe/CUDA interval overlap or general production speedup. Commit {commit}.",
                css_class="small mono",
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    reference_dir = (
        root / "benchmarks" / "reference" / "paged-runtime-olmoe-p310-async-smoke"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=reference_dir / "result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=reference_dir / "sync-vs-async.svg",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed SVG differs from a fresh render",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.reference.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reference must be a JSON object")
    rendered = render_svg(payload)
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            raise ValueError(f"SVG is stale; regenerate {args.output}")
        print(f"Verified {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
