#!/usr/bin/env python3
"""Render the sanitized RTX 6000 Ada paged-runtime study as one SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = (
    ROOT / "benchmarks" / "reference" / "paged-runtime-olmoe-runpod-rtx6000ada-study"
)
DOMAIN_MIN = 0.70
DOMAIN_MAX = 1.70
PLOT_LEFT = 350.0
PLOT_RIGHT = 1235.0


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study result must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported study schema")
    observations = payload.get("observations")
    aggregates = payload.get("condition_aggregates")
    if not isinstance(observations, list) or len(observations) != 72:
        raise ValueError("study must contain 72 pass observations")
    if not isinstance(aggregates, list) or len(aggregates) != 24:
        raise ValueError("study must contain 24 condition aggregates")
    return payload


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _x(value: float) -> float:
    return PLOT_LEFT + (value - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN) * (
        PLOT_RIGHT - PLOT_LEFT
    )


def _text(
    x: float,
    y: float,
    value: object,
    *,
    css_class: str = "label",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}" '
        f'text-anchor="{anchor}">{html.escape(str(value))}</text>'
    )


def _aggregate_index(
    payload: Mapping[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in payload["condition_aggregates"]:
        if not isinstance(raw, dict):
            raise ValueError("aggregate rows must be objects")
        key = (
            raw.get("group"),
            raw.get("workload"),
            raw.get("tokens"),
            raw.get("slots_per_layer"),
            raw.get("cache_condition"),
        )
        if key in result:
            raise ValueError(f"duplicate aggregate: {key}")
        result[key] = raw
    return result


def _axis(y_top: float, y_bottom: float) -> list[str]:
    lines: list[str] = []
    for tick in (0.75, 1.00, 1.25, 1.50, 1.70):
        x = _x(tick)
        kind = "baseline" if tick == 1.0 else "grid"
        lines.append(
            f'<line x1="{x:.1f}" y1="{y_top:.1f}" x2="{x:.1f}" '
            f'y2="{y_bottom:.1f}" class="{kind}"/>'
        )
        lines.append(
            _text(x, y_top - 12, f"{tick:.2f}×", css_class="tick", anchor="middle")
        )
    lines.append(
        _text(
            PLOT_RIGHT,
            y_top - 40,
            "> 1× = async faster",
            css_class="hint",
            anchor="end",
        )
    )
    return lines


def _mark(row: Mapping[str, Any], y: float, condition: str) -> list[str]:
    minimum = _number(row.get("paired_ratio_min"), "minimum ratio")
    median = _number(row.get("paired_ratio_median"), "median ratio")
    maximum = _number(row.get("paired_ratio_max"), "maximum ratio")
    if not DOMAIN_MIN <= minimum <= median <= maximum <= DOMAIN_MAX:
        raise ValueError("ratio interval is outside the fixed chart domain")
    css_class = "cold" if condition == "cold" else "retained"
    lines = [
        (
            f'<line x1="{_x(minimum):.1f}" y1="{y:.1f}" '
            f'x2="{_x(maximum):.1f}" y2="{y:.1f}" '
            f'class="interval {css_class}"/>'
        ),
        (
            f'<line x1="{_x(minimum):.1f}" y1="{y - 5:.1f}" '
            f'x2="{_x(minimum):.1f}" y2="{y + 5:.1f}" '
            f'class="cap {css_class}"/>'
        ),
        (
            f'<line x1="{_x(maximum):.1f}" y1="{y - 5:.1f}" '
            f'x2="{_x(maximum):.1f}" y2="{y + 5:.1f}" '
            f'class="cap {css_class}"/>'
        ),
    ]
    if condition == "cold":
        lines.append(
            f'<circle cx="{_x(median):.1f}" cy="{y:.1f}" r="7" class="point cold"/>'
        )
    else:
        x = _x(median)
        lines.append(
            f'<path d="M {x:.1f} {y - 8:.1f} L {x + 8:.1f} {y:.1f} '
            f'L {x:.1f} {y + 8:.1f} L {x - 8:.1f} {y:.1f} Z" '
            'class="point retained"/>'
        )
    label_x = min(PLOT_RIGHT - 6, _x(maximum) + 12)
    anchor = "end" if label_x >= PLOT_RIGHT - 7 else "start"
    lines.append(
        _text(label_x, y + 5, f"{median:.3f}×", css_class="value", anchor=anchor)
    )
    return lines


def _panel(
    title: str,
    subtitle: str,
    rows: list[tuple[str, str, Mapping[str, Any]]],
    *,
    top: float,
) -> tuple[list[str], float]:
    row_height = 34.0
    y_start = top + 92.0
    bottom = y_start + row_height * (len(rows) - 1) + 26.0
    lines = [
        f'<rect x="45" y="{top:.1f}" width="1230" height="{bottom - top + 30:.1f}" class="panel"/>',
        _text(75, top + 38, title, css_class="section-title"),
        _text(75, top + 66, subtitle, css_class="subtitle"),
        *_axis(y_start - 10, bottom),
    ]
    for index, (label, condition, row) in enumerate(rows):
        y = y_start + index * row_height
        suffix = "empty" if condition == "cold" else "retained"
        lines.append(_text(75, y + 5, f"{label} · {suffix}", css_class="row-label"))
        lines.extend(_mark(row, y, condition))
    return lines, bottom + 55.0


def render(payload: Mapping[str, Any]) -> str:
    index = _aggregate_index(payload)
    workload_rows: list[tuple[str, str, Mapping[str, Any]]] = []
    for workload in (
        "domain_switch",
        "math_reasoning",
        "python_code",
        "systems_en",
        "systems_it",
    ):
        for condition in ("cold", "retained"):
            workload_rows.append(
                (
                    workload,
                    condition,
                    index[("core", workload, 16, 32, condition)],
                )
            )

    length_rows: list[tuple[str, str, Mapping[str, Any]]] = []
    for tokens in (2, 8, 32, 64):
        for condition in ("cold", "retained"):
            length_rows.append(
                (
                    f"{tokens} tokens",
                    condition,
                    index[("length", "python_code", tokens, 32, condition)],
                )
            )

    capacity_rows: list[tuple[str, str, Mapping[str, Any]]] = []
    for slots in (16, 24, 32, 40):
        group = "core" if slots == 32 else "capacity"
        for condition in ("cold", "retained"):
            capacity_rows.append(
                (
                    f"{slots}{'*' if slots == 32 else ''} slots/layer",
                    condition,
                    index[(group, "python_code", 16, slots, condition)],
                )
            )

    lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="1510" '
            'viewBox="0 0 1320 1510" role="img" '
            'aria-labelledby="chart-title chart-desc">'
        ),
        '<title id="chart-title">RTX 6000 Ada synchronous versus asynchronous paged-runtime study</title>',
        '<desc id="chart-desc">Median paired sync over async wall-time ratios with min-to-max intervals across three repetitions. Values above one favor async. Empty-cache results are usually positive, while retained-cache and long-token results include regressions.</desc>',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#E8EEF4}",
        ".bg{fill:#07131A}.panel{fill:#0B1B24;stroke:#294654;stroke-width:1.5}",
        ".title{font-size:32px;font-weight:700}.lede{font-size:17px;fill:#AAB8C2}",
        ".section-title{font-size:21px;font-weight:700}.subtitle{font-size:14px;fill:#93A5B1}",
        ".row-label{font-size:13px;fill:#D6E0E7}.tick{font-size:12px;fill:#8295A2}",
        ".hint{font-size:12px;fill:#AAB8C2}.value{font-size:12px;font-weight:700}",
        ".grid{stroke:#203945;stroke-width:1}.baseline{stroke:#D6E0E7;stroke-width:1.5;stroke-dasharray:5 5}",
        ".interval{stroke-width:4;stroke-linecap:round}.cap{stroke-width:2}",
        ".cold{stroke:#45D4E8}.interval.cold{stroke:#45D4E8}.point.cold{fill:#45D4E8;stroke:#0B1B24;stroke-width:2}",
        ".retained{stroke:#FF8A65}.interval.retained{stroke:#FF8A65}.point.retained{fill:#0B1B24;stroke:#FF8A65;stroke-width:2}",
        ".legend{font-size:13px;fill:#C9D5DC}.note{font-size:13px;fill:#94A8B4}",
        "</style>",
        '<rect width="1320" height="1510" class="bg"/>',
        _text(45, 55, "Paged OLMoE runtime on RTX 6000 Ada", css_class="title"),
        _text(
            45,
            86,
            "Three paired repetitions per case · sync/async wall time · focused ratio scale 0.70–1.70×",
            css_class="lede",
        ),
        '<circle cx="51" cy="119" r="7" class="point cold"/>',
        _text(68, 124, "Empty dynamic expert cache", css_class="legend"),
        '<path d="M 286 111 L 294 119 L 286 127 L 278 119 Z" class="point retained"/>',
        _text(306, 124, "Immediate retained-cache repeat", css_class="legend"),
    ]
    panel, next_top = _panel(
        "Five-workload core comparison",
        "16 teacher-forced tokens · LRU32 per layer · interval is min–max across R1–R3",
        workload_rows,
        top=150,
    )
    lines.extend(panel)
    panel, next_top = _panel(
        "Token-length sensitivity",
        "python_code · LRU32 per layer · longer retained runs expose async overhead",
        length_rows,
        top=next_top,
    )
    lines.extend(panel)
    panel, next_top = _panel(
        "Cache-capacity sensitivity",
        "python_code · 16 tokens · *32-slot point comes from the separate core matrix",
        capacity_rows,
        top=next_top,
    )
    lines.extend(panel)
    lines.extend(
        [
            _text(
                45,
                next_top + 5,
                "Above 1× means async used less wall time; below 1× is a regression. The scale is focused and does not start at zero.",
                css_class="note",
            ),
            _text(
                45,
                next_top + 29,
                "Pinned OLMoE · one GPU · seed 17 · no concurrency · warm host page cache · logical storage bytes, not physical NVMe telemetry.",
                css_class="note",
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REFERENCE_DIR / "result.json")
    parser.add_argument("--output", type=Path, default=REFERENCE_DIR / "study.svg")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    encoded = render(_load(args.input))
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != encoded
        ):
            raise SystemExit(f"rendered chart is stale: {args.output}")
        print(f"Verified {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
