"""Deterministic overlap metrics for CUDA-event timeline intervals.

The caller supplies timestamps in milliseconds elapsed from one shared CUDA
origin event.  This module deliberately does not create CUDA events or make
any scheduling claims: it only summarizes the intervals it is given.  That
makes it suitable for attaching to a benchmark later while keeping the metric
calculation independently testable and reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

TIMELINE_METRICS_KIND = "moevm-cuda-timeline-metrics"
TIMELINE_METRICS_SCHEMA_VERSION = 1
TIMESTAMP_BASIS = "CUDA elapsed milliseconds relative to one shared origin event"


def _finite_milliseconds(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number of milliseconds")
    milliseconds = float(value)
    if not math.isfinite(milliseconds):
        raise ValueError(f"{field_name} must be a finite number of milliseconds")
    return milliseconds


@dataclass(frozen=True, slots=True)
class CudaInterval:
    """One named inclusive-start, exclusive-end interval on a CUDA timeline."""

    name: str
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("interval name must be a non-empty string")
        start_ms = _finite_milliseconds(self.start_ms, "interval start_ms")
        end_ms = _finite_milliseconds(self.end_ms, "interval end_ms")
        if end_ms < start_ms:
            raise ValueError(
                "interval end_ms must be greater than or equal to start_ms"
            )
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds; zero-length event intervals are allowed."""
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, float | str]:
        """Return the stable JSON-ready representation used in the report."""
        return {
            "name": self.name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


IntervalInput: TypeAlias = CudaInterval | Mapping[str, object]


def interval_from_timestamps(name: str, start_ms: float, end_ms: float) -> CudaInterval:
    """Build a validated interval from CUDA elapsed timestamps in milliseconds."""
    return CudaInterval(name=name, start_ms=start_ms, end_ms=end_ms)


def _coerce_interval(value: IntervalInput, lane_name: str) -> CudaInterval:
    if isinstance(value, CudaInterval):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{lane_name} intervals must be CudaInterval objects or mappings"
        )
    required = ("name", "start_ms", "end_ms")
    missing = [field for field in required if field not in value]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{lane_name} interval is missing required field(s): {missing_text}"
        )
    return CudaInterval(
        name=value["name"],  # type: ignore[arg-type]
        start_ms=value["start_ms"],  # type: ignore[arg-type]
        end_ms=value["end_ms"],  # type: ignore[arg-type]
    )


def _canonical_intervals(
    intervals: Iterable[IntervalInput], lane_name: str
) -> tuple[CudaInterval, ...]:
    try:
        parsed = tuple(_coerce_interval(interval, lane_name) for interval in intervals)
    except TypeError as exc:
        raise ValueError(f"{lane_name} intervals must be iterable") from exc
    return tuple(
        sorted(
            parsed,
            key=lambda interval: (interval.start_ms, interval.end_ms, interval.name),
        )
    )


def _merged_ranges(
    intervals: Iterable[CudaInterval],
) -> tuple[tuple[float, float], ...]:
    """Return a canonical union of the positive-duration intervals."""
    ranges: list[tuple[float, float]] = []
    for interval in sorted(
        intervals, key=lambda item: (item.start_ms, item.end_ms, item.name)
    ):
        if interval.duration_ms == 0.0:
            continue
        if not ranges or interval.start_ms > ranges[-1][1]:
            ranges.append((interval.start_ms, interval.end_ms))
        else:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], interval.end_ms))
    return tuple(ranges)


def _range_duration_ms(ranges: Iterable[tuple[float, float]]) -> float:
    return math.fsum(end_ms - start_ms for start_ms, end_ms in ranges)


def _nonnegative_remainder(total_ms: float, covered_ms: float) -> float:
    """Subtract durations without exposing a negative floating-point artifact."""
    return max(0.0, total_ms - covered_ms)


def _range_overlap_ms(
    left: tuple[tuple[float, float], ...], right: tuple[tuple[float, float], ...]
) -> float:
    """Measure the intersection of two already-merged interval sets."""
    left_index = 0
    right_index = 0
    overlaps: list[float] = []
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        if overlap_end > overlap_start:
            overlaps.append(overlap_end - overlap_start)
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return math.fsum(overlaps)


def _lane_summary(
    intervals: tuple[CudaInterval, ...], merged_ranges: tuple[tuple[float, float], ...]
) -> dict[str, float | int | None]:
    raw_duration_ms = math.fsum(interval.duration_ms for interval in intervals)
    active_duration_ms = _range_duration_ms(merged_ranges)
    return {
        "interval_count": len(intervals),
        "start_ms": intervals[0].start_ms if intervals else None,
        # Intervals are ordered by start time, so the last submitted interval
        # can be nested inside an earlier, longer interval.  Report the actual
        # lane endpoint rather than assuming end times share that ordering.
        "end_ms": max(interval.end_ms for interval in intervals) if intervals else None,
        "raw_duration_ms": raw_duration_ms,
        "active_duration_ms": active_duration_ms,
        "intra_lane_overlap_ms": _nonnegative_remainder(
            raw_duration_ms, active_duration_ms
        ),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarize_cuda_timeline(
    *,
    transfers: Iterable[IntervalInput],
    compute: Iterable[IntervalInput],
) -> dict[str, Any]:
    """Summarize transfer/compute overlap from a shared-origin CUDA timeline.

    ``transfers`` and ``compute`` may contain :class:`CudaInterval` objects or
    mappings with ``name``, ``start_ms``, and ``end_ms`` fields.  Input order is
    intentionally ignored: interval records and all derived values are sorted
    before aggregation, so equivalent timestamp sets return the same report.

    ``raw_duration_ms`` sums every submitted interval.  ``active_duration_ms``
    unions intervals within each lane first, which prevents overlapping copies
    or kernels in the same lane from being counted twice.  The overlap metrics
    are calculated from those two unioned lanes.
    """
    transfer_intervals = _canonical_intervals(transfers, "transfer")
    compute_intervals = _canonical_intervals(compute, "compute")
    transfer_ranges = _merged_ranges(transfer_intervals)
    compute_ranges = _merged_ranges(compute_intervals)
    transfer_summary = _lane_summary(transfer_intervals, transfer_ranges)
    compute_summary = _lane_summary(compute_intervals, compute_ranges)

    all_intervals = transfer_intervals + compute_intervals
    if all_intervals:
        timeline_start_ms = min(interval.start_ms for interval in all_intervals)
        timeline_end_ms = max(interval.end_ms for interval in all_intervals)
    else:
        timeline_start_ms = None
        timeline_end_ms = None
    timeline_span_ms = (
        timeline_end_ms - timeline_start_ms
        if timeline_start_ms is not None and timeline_end_ms is not None
        else 0.0
    )

    transfer_active_ms = float(transfer_summary["active_duration_ms"])
    compute_active_ms = float(compute_summary["active_duration_ms"])
    overlap_ms = _range_overlap_ms(transfer_ranges, compute_ranges)
    combined_ranges = _merged_ranges(transfer_intervals + compute_intervals)
    combined_active_ms = _range_duration_ms(combined_ranges)
    serial_active_ms = transfer_active_ms + compute_active_ms

    return {
        "schema_version": TIMELINE_METRICS_SCHEMA_VERSION,
        "kind": TIMELINE_METRICS_KIND,
        "timestamp_basis": TIMESTAMP_BASIS,
        "intervals": {
            "transfer": [interval.to_dict() for interval in transfer_intervals],
            "compute": [interval.to_dict() for interval in compute_intervals],
        },
        "timeline": {
            "start_ms": timeline_start_ms,
            "end_ms": timeline_end_ms,
            "span_ms": timeline_span_ms,
            "combined_active_duration_ms": combined_active_ms,
        },
        "transfer": transfer_summary,
        "compute": compute_summary,
        "overlap": {
            "duration_ms": overlap_ms,
            "transfer_overlap_fraction": _ratio(overlap_ms, transfer_active_ms),
            "compute_overlap_fraction": _ratio(overlap_ms, compute_active_ms),
            "transfer_hidden_by_compute_ms": overlap_ms,
            "transfer_exposed_ms": _nonnegative_remainder(
                transfer_active_ms, overlap_ms
            ),
            "compute_exposed_ms": _nonnegative_remainder(compute_active_ms, overlap_ms),
            "serial_active_duration_ms": serial_active_ms,
            "active_duration_saved_by_overlap_ms": _nonnegative_remainder(
                serial_active_ms, combined_active_ms
            ),
        },
    }


__all__ = [
    "TIMELINE_METRICS_KIND",
    "TIMELINE_METRICS_SCHEMA_VERSION",
    "TIMESTAMP_BASIS",
    "CudaInterval",
    "IntervalInput",
    "interval_from_timestamps",
    "summarize_cuda_timeline",
]
