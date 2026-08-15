#!/usr/bin/env python3
"""Render one opt-in CUDA H2D/expert-compute capture as an accessible SVG.

The input is the JSON emitted by ``benchmark_paged_olmoe.py`` with
``--cuda-overlap-telemetry``.  CUDA event timestamps are only comparable inside
one model invocation, so this renderer deliberately requires one pass and one
call rather than combining prefill and decode captures.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PASS_NAMES = ("cold_expert_cache", "repeat_retained_expert_cache")
_WIDTH = 1280
_PLOT_LEFT = 230.0
_PLOT_RIGHT = 1230.0
_ROW_HEIGHT = 34.0


@dataclass(frozen=True, slots=True)
class CallSelection:
    """One model invocation within a benchmark pass."""

    kind: str
    decode_index: int | None = None

    @property
    def cli_value(self) -> str:
        if self.kind == "prefill":
            return "prefill"
        if self.decode_index is None:  # pragma: no cover - construction invariant
            raise ValueError("decode selection is missing its index")
        return f"decode:{self.decode_index}"

    @property
    def display_name(self) -> str:
        if self.kind == "prefill":
            return "prefill"
        if self.decode_index is None:  # pragma: no cover - construction invariant
            raise ValueError("decode selection is missing its index")
        return f"decode token {self.decode_index}"


@dataclass(frozen=True, slots=True)
class TimelineSpan:
    """A validated interval from one shared-origin CUDA event capture."""

    lane: str
    name: str
    start_ms: float
    end_ms: float
    layer: int | None
    expert: int | None
    sequence: int | None

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def direct_label(self) -> str:
        if self.layer is not None and self.expert is not None:
            return f"L{self.layer} · E{self.expert}"
        return self.name

    @property
    def accessible_label(self) -> str:
        return (
            f"{self.lane}: {self.direct_label}, "
            f"{_format_ms(self.start_ms)} to {_format_ms(self.end_ms)}"
        )


@dataclass(frozen=True, slots=True)
class SelectedTimeline:
    """The selected call and its self-contained CUDA-event payload."""

    pass_name: str
    selection: CallSelection
    status: str
    spans: tuple[TimelineSpan, ...]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer when present")
    return value


def parse_call(value: str) -> CallSelection:
    """Parse ``prefill`` or a benchmark ``per_token[].index`` selector."""

    if value == "prefill":
        return CallSelection(kind="prefill")
    prefix = "decode:"
    index_text = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not index_text.isascii()
        or not index_text.isdigit()
    ):
        raise argparse.ArgumentTypeError(
            "--call must be prefill or decode:<non-negative token index>"
        )
    return CallSelection(kind="decode", decode_index=int(index_text))


def load_benchmark(path: Path) -> Mapping[str, Any]:
    """Load the benchmark result without accepting a non-object JSON value."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"input is not valid JSON: {exc.msg}") from exc
    return _mapping(payload, "input benchmark")


def _timeline_span(raw: object, index: int) -> TimelineSpan:
    item = _mapping(raw, f"cuda_event_timeline.spans[{index}]")
    lane = item.get("lane")
    if lane not in {"h2d", "expert_compute"}:
        raise ValueError(
            f"cuda_event_timeline.spans[{index}].lane must be h2d or expert_compute"
        )
    start_ms = _finite_number(
        item.get("start_ms"), f"cuda_event_timeline.spans[{index}].start_ms"
    )
    end_ms = _finite_number(
        item.get("end_ms"), f"cuda_event_timeline.spans[{index}].end_ms"
    )
    if end_ms < start_ms:
        raise ValueError(f"cuda_event_timeline.spans[{index}].end_ms precedes start_ms")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        name = f"{lane}:{index}"
    return TimelineSpan(
        lane=lane,
        name=name,
        start_ms=start_ms,
        end_ms=end_ms,
        layer=_optional_integer(
            item.get("layer"), f"cuda_event_timeline.spans[{index}].layer"
        ),
        expert=_optional_integer(
            item.get("expert"), f"cuda_event_timeline.spans[{index}].expert"
        ),
        sequence=_optional_integer(
            item.get("sequence"), f"cuda_event_timeline.spans[{index}].sequence"
        ),
    )


def _selected_call_payload(
    payload: Mapping[str, Any],
    pass_name: str,
    selection: CallSelection,
) -> Mapping[str, Any]:
    passes = _mapping(payload.get("passes"), "input benchmark.passes")
    if pass_name not in PASS_NAMES:
        raise ValueError(f"unsupported pass: {pass_name}")
    pass_payload = _mapping(passes.get(pass_name), f"passes.{pass_name}")
    if selection.kind == "prefill":
        return _mapping(pass_payload.get("prefill"), f"passes.{pass_name}.prefill")

    decode = _mapping(pass_payload.get("decode"), f"passes.{pass_name}.decode")
    records = decode.get("per_token")
    if not isinstance(records, list):
        raise ValueError(f"passes.{pass_name}.decode.per_token must be an array")
    matching: list[Mapping[str, Any]] = []
    available: list[int] = []
    for position, raw in enumerate(records):
        record = _mapping(raw, f"passes.{pass_name}.decode.per_token[{position}]")
        index = _optional_integer(
            record.get("index"),
            f"passes.{pass_name}.decode.per_token[{position}].index",
        )
        if index is None or index < 0:
            raise ValueError(
                f"passes.{pass_name}.decode.per_token[{position}].index "
                "must be a non-negative integer"
            )
        available.append(index)
        if index == selection.decode_index:
            matching.append(record)
    if len(matching) != 1:
        available_text = ", ".join(str(index) for index in sorted(available))
        if not available_text:
            available_text = "none"
        raise ValueError(
            f"decode call {selection.cli_value} is unavailable in {pass_name}; "
            f"available decode indexes: {available_text}"
        )
    return matching[0]


def select_timeline(
    payload: Mapping[str, Any],
    *,
    pass_name: str,
    selection: CallSelection,
) -> SelectedTimeline:
    """Select and validate one same-origin capture from a benchmark JSON."""

    call = _selected_call_payload(payload, pass_name, selection)
    timeline = _mapping(
        call.get("cuda_event_timeline"),
        (
            f"passes.{pass_name}.{selection.cli_value}.cuda_event_timeline "
            "(run the benchmark with --cuda-overlap-telemetry)"
        ),
    )
    if timeline.get("schema_version") != 1:
        raise ValueError("cuda_event_timeline.schema_version must be 1")
    if timeline.get("complete") is not True:
        raise ValueError("cuda_event_timeline must be complete before rendering")
    if timeline.get("method") != "cuda_events_v1":
        raise ValueError("cuda_event_timeline.method must be cuda_events_v1")
    if timeline.get("scope") != "paged_expert_h2d_vs_expert_compute":
        raise ValueError("cuda_event_timeline scope is not paged-expert H2D/compute")
    if timeline.get("unit") != "milliseconds":
        raise ValueError("cuda_event_timeline.unit must be milliseconds")
    status = timeline.get("status")
    if status not in {"measured", "not_applicable"}:
        raise ValueError(
            "cuda_event_timeline.status must be measured or not_applicable"
        )
    spans = timeline.get("spans")
    if not isinstance(spans, list):
        raise ValueError("cuda_event_timeline.spans must be an array")
    return SelectedTimeline(
        pass_name=pass_name,
        selection=selection,
        status=status,
        spans=tuple(_timeline_span(raw, index) for index, raw in enumerate(spans)),
    )


def _format_ms(value: float) -> str:
    absolute = abs(value)
    if absolute >= 100.0:
        return f"{value:.1f} ms"
    if absolute >= 1.0:
        return f"{value:.3f} ms"
    if absolute >= 0.01:
        return f"{value:.4f} ms"
    return f"{value:.6f} ms"


def _merged_ranges(spans: Iterable[TimelineSpan]) -> tuple[tuple[float, float], ...]:
    ranges: list[tuple[float, float]] = []
    for span in sorted(spans, key=lambda item: (item.start_ms, item.end_ms, item.name)):
        if span.duration_ms == 0.0:
            continue
        if not ranges or span.start_ms > ranges[-1][1]:
            ranges.append((span.start_ms, span.end_ms))
        else:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], span.end_ms))
    return tuple(ranges)


def _overlap_ranges(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            result.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return tuple(result)


def _range_duration(ranges: Iterable[tuple[float, float]]) -> float:
    return math.fsum(end - start for start, end in ranges)


def _pack_rows(spans: Iterable[TimelineSpan]) -> tuple[tuple[TimelineSpan, ...], ...]:
    """Place overlapping intervals on separate compact rows within a lane."""

    rows: list[list[TimelineSpan]] = []
    row_end_ms: list[float] = []
    for span in sorted(
        spans,
        key=lambda item: (
            item.start_ms,
            item.end_ms,
            item.sequence if item.sequence is not None else -1,
            item.name,
        ),
    ):
        row_index = next(
            (
                index
                for index, row_end in enumerate(row_end_ms)
                if span.start_ms >= row_end
            ),
            None,
        )
        if row_index is None:
            row_index = len(rows)
            rows.append([])
            row_end_ms.append(span.end_ms)
        else:
            row_end_ms[row_index] = max(row_end_ms[row_index], span.end_ms)
        rows[row_index].append(span)
    return tuple(tuple(row) for row in rows) or ((),)


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
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def _pass_display_name(pass_name: str) -> str:
    return (
        "cold expert cache"
        if pass_name == "cold_expert_cache"
        else "repeat retained expert cache"
    )


def render_svg(timeline: SelectedTimeline) -> str:
    """Render a deterministic, standalone SVG for one selected CUDA call."""

    h2d_spans = tuple(span for span in timeline.spans if span.lane == "h2d")
    compute_spans = tuple(
        span for span in timeline.spans if span.lane == "expert_compute"
    )
    h2d_rows = _pack_rows(h2d_spans)
    compute_rows = _pack_rows(compute_spans)
    all_spans = h2d_spans + compute_spans
    if all_spans:
        min_ms = min(span.start_ms for span in all_spans)
        max_ms = max(span.end_ms for span in all_spans)
    else:
        min_ms = 0.0
        max_ms = 1.0
    actual_span_ms = max_ms - min_ms
    scale_span_ms = max(actual_span_ms, 0.001)

    h2d_ranges = _merged_ranges(h2d_spans)
    compute_ranges = _merged_ranges(compute_spans)
    overlap_ranges = _overlap_ranges(h2d_ranges, compute_ranges)
    overlap_ms = _range_duration(overlap_ranges)
    h2d_active_ms = _range_duration(h2d_ranges)
    compute_active_ms = _range_duration(compute_ranges)

    h2d_top = 158.0
    h2d_height = 44.0 + len(h2d_rows) * _ROW_HEIGHT + 16.0
    compute_top = h2d_top + h2d_height + 26.0
    compute_height = 44.0 + len(compute_rows) * _ROW_HEIGHT + 16.0
    body_top = h2d_top + 43.0
    body_bottom = compute_top + compute_height - 16.0
    summary_top = compute_top + compute_height + 32.0
    height = int(summary_top + 164.0)

    def x_position(value: float) -> float:
        return _PLOT_LEFT + (value - min_ms) / scale_span_ms * (
            _PLOT_RIGHT - _PLOT_LEFT
        )

    title = (
        "CUDA H2D ↔ expert compute overlap · "
        f"{_pass_display_name(timeline.pass_name)} · {timeline.selection.display_name}"
    )
    description = (
        f"One shared-origin CUDA-event timeline from the {timeline.pass_name} pass, "
        f"{timeline.selection.display_name}. It contains {len(h2d_spans)} H2D spans "
        f"and {len(compute_spans)} expert-compute spans. Highlighted vertical bands "
        f"show {overlap_ms:.6f} milliseconds where the two active lanes overlap. "
        "This instrumentation does not prove physical NVMe activity or end-to-end speedup."
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" '
            f'height="{height}" viewBox="0 0 {_WIDTH} {height}" role="img" '
            'aria-labelledby="timeline-title timeline-desc">'
        ),
        f'<title id="timeline-title">{html.escape(title)}</title>',
        f'<desc id="timeline-desc">{html.escape(description)}</desc>',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#EAF1F7}",
        ".bg{fill:#07131A}.panel{fill:#0D202A;stroke:#274552;stroke-width:1}",
        ".title{font-size:25px;font-weight:700}.subtitle{font-size:14px;fill:#AABBC6}",
        ".lane{font-size:16px;font-weight:700}.lane-detail{font-size:12px;fill:#9FB2BE}",
        ".axis{font-size:11px;fill:#93A8B4}.summary{font-size:14px;font-weight:650}",
        ".note{font-size:12px;fill:#A4B5C0}.bar-label{font-size:11px;font-weight:700;fill:#07131A}",
        ".grid{stroke:#284752;stroke-width:1}.axis-line{stroke:#86A1AF;stroke-width:1}",
        ".overlap{fill:#8EEA9B;fill-opacity:.20}.h2d-bar{fill:#58D4E8;stroke:#C7FAFF;stroke-width:.7}",
        ".compute-bar{fill:#F9A66C;stroke:#FFE0CB;stroke-width:.7}.empty{font-size:13px;fill:#AABBC6}",
        "</style>",
        f'<rect width="{_WIDTH}" height="{height}" class="bg"/>',
        _text(48, 56, "CUDA overlap timeline", css_class="title"),
        _text(
            48,
            82,
            f"{_pass_display_name(timeline.pass_name)} · {timeline.selection.display_name} · CUDA events in one model call",
            css_class="subtitle",
        ),
        _text(
            48,
            108,
            "H2D and compute timestamps share this call's origin; the renderer never combines origins across calls.",
            css_class="subtitle",
        ),
        f'<rect x="42" y="{h2d_top:.1f}" width="1196" height="{h2d_height:.1f}" rx="10" class="panel"/>',
        f'<rect x="42" y="{compute_top:.1f}" width="1196" height="{compute_height:.1f}" rx="10" class="panel"/>',
    ]

    for tick in range(5):
        value = min_ms + scale_span_ms * tick / 4
        x = x_position(value)
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{body_top:.1f}" x2="{x:.1f}" y2="{body_bottom:.1f}" class="grid"/>',
                _text(
                    x,
                    h2d_top + 33,
                    _format_ms(value),
                    css_class="axis",
                    anchor="middle",
                ),
            ]
        )

    for start_ms, end_ms in overlap_ranges:
        start_x = x_position(start_ms)
        width = max(1.0, x_position(end_ms) - start_x)
        lines.append(
            f'<rect x="{start_x:.1f}" y="{body_top:.1f}" width="{width:.1f}" '
            f'height="{body_bottom - body_top:.1f}" class="overlap">'
            f"<title>{html.escape(f'Overlap: {_format_ms(end_ms - start_ms)}')}</title></rect>"
        )

    lines.extend(
        [
            _text(66, h2d_top + 27, "H2D transfer", css_class="lane"),
            _text(66, h2d_top + 46, "host → GPU", css_class="lane-detail"),
            _text(66, compute_top + 27, "Expert compute", css_class="lane"),
            _text(
                66, compute_top + 46, "routed expert kernels", css_class="lane-detail"
            ),
            f'<line x1="{_PLOT_LEFT:.1f}" y1="{body_top:.1f}" x2="{_PLOT_RIGHT:.1f}" y2="{body_top:.1f}" class="axis-line"/>',
            f'<line x1="{_PLOT_LEFT:.1f}" y1="{compute_top + 43:.1f}" x2="{_PLOT_RIGHT:.1f}" y2="{compute_top + 43:.1f}" class="axis-line"/>',
        ]
    )

    def append_lane(
        rows: tuple[tuple[TimelineSpan, ...], ...], *, top: float, css_class: str
    ) -> None:
        for row_index, row in enumerate(rows):
            y = top + 51.0 + row_index * _ROW_HEIGHT
            for span in row:
                x = x_position(span.start_ms)
                width = max(2.0, x_position(span.end_ms) - x)
                label = span.direct_label
                lines.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
                    f'height="22" rx="4" class="{css_class}">'
                    f"<title>{html.escape(span.accessible_label)}</title></rect>"
                )
                if width >= 55.0:
                    lines.append(
                        _text(
                            x + 5.0,
                            y + 15.0,
                            label,
                            css_class="bar-label",
                        )
                    )

    append_lane(h2d_rows, top=h2d_top, css_class="h2d-bar")
    append_lane(compute_rows, top=compute_top, css_class="compute-bar")

    if not all_spans:
        lines.append(
            _text(
                (_PLOT_LEFT + _PLOT_RIGHT) / 2,
                (body_top + body_bottom) / 2,
                "No paged-expert H2D or expert-compute intervals were recorded for this call.",
                css_class="empty",
                anchor="middle",
            )
        )

    overlap_fraction = None if h2d_active_ms == 0.0 else overlap_ms / h2d_active_ms
    fraction_text = (
        "n/a (no H2D active time)"
        if overlap_fraction is None
        else f"{overlap_fraction * 100:.1f}% of H2D active time"
    )
    lines.extend(
        [
            f'<rect x="42" y="{summary_top:.1f}" width="1196" height="132" rx="10" class="panel"/>',
            _text(
                66,
                summary_top + 30,
                f"Overlap highlighted: {_format_ms(overlap_ms)} · {fraction_text}",
                css_class="summary",
            ),
            _text(
                66,
                summary_top + 57,
                f"H2D: {len(h2d_spans)} spans / {_format_ms(h2d_active_ms)} active · Expert compute: {len(compute_spans)} spans / {_format_ms(compute_active_ms)} active",
                css_class="note",
            ),
            _text(
                66,
                summary_top + 86,
                "Caveat: CUDA-event intervals cover paged-expert H2D and expert compute only; they do not establish physical NVMe activity, page-cache state, or an end-to-end speedup.",
                css_class="note",
            ),
            _text(
                66,
                summary_top + 109,
                "Use this to inspect one call. Compare wall time only between paired runs collected with the same telemetry setting.",
                css_class="note",
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def write_svg_create_only(path: Path, contents: str) -> None:
    """Write ``path`` once, never replacing an existing benchmark artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(contents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Benchmark JSON input."
    )
    parser.add_argument(
        "--pass",
        dest="pass_name",
        required=True,
        choices=PASS_NAMES,
        help="Benchmark cache pass containing the selected model call.",
    )
    parser.add_argument(
        "--call",
        dest="selection",
        required=True,
        type=parse_call,
        help="prefill, or decode:<per_token index> (the first decode is normally decode:1).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Create-only SVG output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_benchmark(args.input)
    selected = select_timeline(
        payload,
        pass_name=args.pass_name,
        selection=args.selection,
    )
    write_svg_create_only(args.output, render_svg(selected))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
