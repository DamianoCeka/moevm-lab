"""Fail-closed comparison of one sync/async paged-runtime benchmark pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moevm.timeline_metrics import CudaInterval, summarize_cuda_timeline

PASS_NAMES = ("cold_expert_cache", "repeat_retained_expert_cache")
PRIMITIVES = (
    "requests",
    "hits",
    "misses",
    "evictions",
    "storage_loads",
    "transfer_loads",
    "storage_bytes",
    "host_to_device_bytes",
    "coalesced_requests",
    "admission_rejections",
    "storage_failures",
    "transfer_failures",
)
_CACHE_PRIMITIVES = PRIMITIVES[:8]
_ZERO_COUNTERS = PRIMITIVES[8:]
ADDITIVE_PRIMITIVES = PRIMITIVES
ZERO_SAFETY_COUNTERS = (
    "coalesced_requests",
    "admission_rejections",
    "storage_failures",
    "transfer_failures",
)
NON_NEGATIVE_TIME_FIELDS = (
    "storage_seconds",
    "transfer_seconds",
    "forward_seconds",
    "storage_queue_seconds",
    "demand_wait_seconds",
)
MAX_EXTRA_ASYNC_PEAK_VRAM_BYTES = 64 * 1024 * 1024
RUNTIME_IDENTITY_FIELDS = (
    "device",
    "device_uuid",
    "device_name",
    "policy",
    "capacity_scope",
    "hotset_json",
    "hotset_sha256",
    "protected_hot_per_layer",
    "cuda_overlap_telemetry",
)
PASS_IDENTITY_FIELDS = (
    "teacher_forced",
    "generated_token_count",
    "generated_ids",
    "fed_token_ids",
    "reference_prediction",
)
REFERENCE_IDENTITY_FIELDS = (
    "available",
    "first_mismatch_index",
    "generated_token_ids",
    "matched",
    "matched_tokens",
    "mode",
    "sha256",
    "source_generated_token_count",
    "temperature",
    "total_tokens",
)
_CUDA_TIMELINE_METHOD = "cuda_events_v1"
_CUDA_TIMELINE_SCOPE = "paged_expert_h2d_vs_expert_compute"
_CUDA_TIMELINE_UNIT = "milliseconds"
# The outer benchmark report remains schema v1.  The nested CUDA telemetry
# contract is v2 because v1 could not prove H2D-span coverage.
_CUDA_TIMELINE_SCHEMA_VERSION = 2
_LEGACY_CUDA_TIMELINE_SCHEMA_VERSION = 1
_CUDA_TIMELINE_AGGREGATION = (
    "Summed per-model-call CUDA-event lane summaries; timestamps from different "
    "model calls are not unioned."
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _telemetry_diagnostic_value(value: object, *, limit: int) -> str:
    """Render untrusted capture diagnostics without dumping arbitrary payloads."""

    if value is None:
        return "null"
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    normalized = " ".join(value.split())
    if len(normalized) > limit:
        normalized = f"{normalized[: limit - 1]}…"
    return repr(normalized)


def _incomplete_timeline_detail(timeline: dict[str, Any]) -> str:
    """Return the runtime's bounded incomplete-capture explanation."""

    return (
        " ("
        f"status={_telemetry_diagnostic_value(timeline.get('status'), limit=64)}; "
        f"reason={_telemetry_diagnostic_value(timeline.get('reason'), limit=240)}"
        ")"
    )


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    minimum_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not minimum_ok:
        comparison = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {comparison}")
    return result


def _require_equal(left: object, right: object, name: str) -> None:
    if left != right:
        raise ValueError(f"sync/async mismatch at {name}: {left!r} != {right!r}")


def _metrics(payload: object, name: str, expert_bytes: int) -> dict[str, int]:
    values = _mapping(payload, name)
    result = {
        field: _integer(values.get(field), f"{name}.{field}") for field in PRIMITIVES
    }
    if result["requests"] != result["hits"] + result["misses"]:
        raise ValueError(f"{name}: requests must equal hits + misses")
    if result["storage_bytes"] != result["storage_loads"] * expert_bytes:
        raise ValueError(f"{name}: storage byte/load invariant failed")
    if result["host_to_device_bytes"] != result["transfer_loads"] * expert_bytes:
        raise ValueError(f"{name}: H2D byte/load invariant failed")
    if result["coalesced_requests"] == 0 and (
        result["misses"] != result["storage_loads"]
        or result["misses"] != result["transfer_loads"]
    ):
        raise ValueError(f"{name}: zero-coalescing miss/load invariant failed")
    for field in ZERO_SAFETY_COUNTERS:
        if result[field] != 0:
            raise ValueError(f"{name}.{field} must be zero for a comparable pair")
    expected_hit_rate = (
        result["hits"] / result["requests"] if result["requests"] else 0.0
    )
    hit_rate = _number(values.get("hit_rate"), f"{name}.hit_rate")
    if not math.isclose(hit_rate, expected_hit_rate, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{name}: hit_rate is inconsistent with hits/requests")
    for field in NON_NEGATIVE_TIME_FIELDS:
        _number(values.get(field), f"{name}.{field}")
    _integer(values.get("staging_waits"), f"{name}.staging_waits")
    return result


def _cuda_overlap_requested(runtime: dict[str, Any], mode: str) -> bool:
    """Return whether a report must carry strict CUDA-event telemetry.

    Reports from before the opt-in feature intentionally omit this object and
    all telemetry result fields; those reports remain comparable under the
    pre-existing counter contract.  Once a result claims that telemetry was
    requested, every recorded call must be complete and internally
    reproducible from its raw CUDA-event spans.
    """

    raw = runtime.get("cuda_overlap_telemetry")
    if raw is None:
        return False
    config = _mapping(raw, f"{mode}.runtime.cuda_overlap_telemetry")
    requested = config.get("requested")
    if not isinstance(requested, bool):
        raise ValueError(
            f"{mode}.runtime.cuda_overlap_telemetry.requested must be bool"
        )
    expected_method = _CUDA_TIMELINE_METHOD if requested else None
    expected_scope = _CUDA_TIMELINE_SCOPE if requested else None
    if config.get("method") != expected_method:
        raise ValueError(f"{mode}.runtime.cuda_overlap_telemetry.method is invalid")
    if config.get("scope") != expected_scope:
        raise ValueError(f"{mode}.runtime.cuda_overlap_telemetry.scope is invalid")
    return requested


def _require_covered_timeline_schema_version(value: object, name: str) -> None:
    """Reject v1 telemetry explicitly rather than treating it as covered."""

    schema_version = _integer(value, name)
    if schema_version == _LEGACY_CUDA_TIMELINE_SCHEMA_VERSION:
        raise ValueError(
            f"{name}=1 is legacy-unverified; covered CUDA telemetry requires "
            f"schema_version={_CUDA_TIMELINE_SCHEMA_VERSION}"
        )
    if schema_version != _CUDA_TIMELINE_SCHEMA_VERSION:
        raise ValueError(f"{name} must be {_CUDA_TIMELINE_SCHEMA_VERSION}")


def _timeline_summary_equal(actual: object, expected: object) -> bool:
    """Compare derived JSON values without accepting bools as integer counts."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return False
        return all(
            _timeline_summary_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _timeline_summary_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, int)
            and actual == expected
        )
    return type(actual) is type(expected) and actual == expected


def _timeline_span(raw: object, name: str) -> tuple[str, int, CudaInterval]:
    span = _mapping(raw, name)
    lane = span.get("lane")
    if lane not in {"h2d", "expert_compute"}:
        raise ValueError(f"{name}.lane must be h2d or expert_compute")
    sequence = _integer(span.get("sequence"), f"{name}.sequence")
    layer = _integer(span.get("layer"), f"{name}.layer")
    expert = _integer(span.get("expert"), f"{name}.expert")
    span_name = span.get("name")
    expected_name = f"{lane}:{sequence}:L{layer}:E{expert}"
    if span_name != expected_name:
        raise ValueError(f"{name}.name must equal {expected_name!r}")
    start_ms = _number(span.get("start_ms"), f"{name}.start_ms")
    end_ms = _number(span.get("end_ms"), f"{name}.end_ms")
    if end_ms < start_ms:
        raise ValueError(f"{name}.end_ms must be greater than or equal to start_ms")
    duration_ms = _number(span.get("duration_ms"), f"{name}.duration_ms")
    if not math.isclose(duration_ms, end_ms - start_ms, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name}.duration_ms is inconsistent with its endpoints")
    return (
        lane,
        sequence,
        CudaInterval(
            name=expected_name,
            start_ms=start_ms,
            end_ms=end_ms,
        ),
    )


def _timeline_coverage(
    timeline: dict[str, Any],
    *,
    h2d_span_count: int,
    name: str,
) -> None:
    """Require cache-transfer accounting to cover every raw H2D event."""

    coverage = _mapping(timeline.get("coverage"), f"{name}.coverage")
    cache_transfer_loads_delta = _integer(
        coverage.get("cache_transfer_loads_delta"),
        f"{name}.coverage.cache_transfer_loads_delta",
    )
    reported_h2d_span_count = _integer(
        coverage.get("h2d_span_count"),
        f"{name}.coverage.h2d_span_count",
    )
    if reported_h2d_span_count != h2d_span_count:
        raise ValueError(
            f"{name}.coverage.h2d_span_count is inconsistent with raw H2D spans"
        )
    if cache_transfer_loads_delta != h2d_span_count:
        raise ValueError(
            f"{name}.coverage.cache_transfer_loads_delta is inconsistent with raw H2D spans"
        )


def _timeline_call(raw: object, name: str) -> dict[str, Any]:
    timeline = _mapping(raw, name)
    _require_covered_timeline_schema_version(
        timeline.get("schema_version"),
        f"{name}.schema_version",
    )
    if timeline.get("complete") is not True:
        raise ValueError(
            f"{name} must be complete{_incomplete_timeline_detail(timeline)}"
        )
    if timeline.get("method") != _CUDA_TIMELINE_METHOD:
        raise ValueError(f"{name}.method is invalid")
    if timeline.get("scope") != _CUDA_TIMELINE_SCOPE:
        raise ValueError(f"{name}.scope is invalid")
    if timeline.get("unit") != _CUDA_TIMELINE_UNIT:
        raise ValueError(f"{name}.unit is invalid")
    status = timeline.get("status")
    if status not in {"measured", "not_applicable"}:
        raise ValueError(f"{name}.status must be measured or not_applicable")
    raw_spans = timeline.get("spans")
    if not isinstance(raw_spans, list):
        raise ValueError(f"{name}.spans must be an array")
    transfers: list[CudaInterval] = []
    compute: list[CudaInterval] = []
    sequences: set[int] = set()
    for index, raw_span in enumerate(raw_spans):
        lane, sequence, interval = _timeline_span(raw_span, f"{name}.spans[{index}]")
        if sequence in sequences:
            raise ValueError(f"{name}.spans[{index}].sequence must be unique")
        sequences.add(sequence)
        if lane == "h2d":
            transfers.append(interval)
        else:
            compute.append(interval)

    expected_summary = summarize_cuda_timeline(
        transfers=transfers,
        compute=compute,
    )
    expected_summary.pop("intervals")
    if not _timeline_summary_equal(timeline.get("summary"), expected_summary):
        raise ValueError(f"{name}.summary is not derived from its raw spans")
    _timeline_coverage(
        timeline,
        h2d_span_count=len(transfers),
        name=name,
    )

    has_both_lanes = bool(transfers) and bool(compute)
    reason = timeline.get("reason")
    if status == "measured":
        if not has_both_lanes:
            raise ValueError(f"{name}.status=measured requires both event lanes")
        if reason is not None:
            raise ValueError(f"{name}.reason must be null when status=measured")
    else:
        if has_both_lanes:
            raise ValueError(f"{name}.status=not_applicable has both event lanes")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{name}.reason is required when status=not_applicable")
    return {
        "status": status,
        "summary": expected_summary,
        "reason": reason,
    }


def _timeline_number_equal(
    actual: object,
    expected: float,
    name: str,
) -> None:
    value = _number(actual, name)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} is inconsistent with model-call timelines")


def _timeline_optional_ratio_equal(
    actual: object,
    expected: float | None,
    name: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{name} must be null when its denominator is zero")
        return
    _timeline_number_equal(actual, expected, name)


def _pass_cuda_overlap(
    raw: object,
    calls: list[dict[str, Any]],
    name: str,
) -> None:
    if not calls:
        raise ValueError(f"{name} requires at least one model-call timeline")
    aggregate = _mapping(raw, name)
    _require_covered_timeline_schema_version(
        aggregate.get("schema_version"),
        f"{name}.schema_version",
    )
    if aggregate.get("method") != _CUDA_TIMELINE_METHOD:
        raise ValueError(f"{name}.method is invalid")
    if aggregate.get("scope") != _CUDA_TIMELINE_SCOPE:
        raise ValueError(f"{name}.scope is invalid")
    if aggregate.get("unit") != _CUDA_TIMELINE_UNIT:
        raise ValueError(f"{name}.unit is invalid")
    if aggregate.get("aggregation") != _CUDA_TIMELINE_AGGREGATION:
        raise ValueError(f"{name}.aggregation is invalid")

    expected_h2d_intervals = sum(
        int(call["summary"]["transfer"]["interval_count"]) for call in calls
    )
    expected_compute_intervals = sum(
        int(call["summary"]["compute"]["interval_count"]) for call in calls
    )
    expected_h2d_active = math.fsum(
        float(call["summary"]["transfer"]["active_duration_ms"]) for call in calls
    )
    expected_compute_active = math.fsum(
        float(call["summary"]["compute"]["active_duration_ms"]) for call in calls
    )
    expected_overlap = math.fsum(
        float(call["summary"]["overlap"]["duration_ms"]) for call in calls
    )
    expected_saved = math.fsum(
        float(call["summary"]["overlap"]["active_duration_saved_by_overlap_ms"])
        for call in calls
    )
    measured_calls = sum(call["status"] == "measured" for call in calls)
    expected_status = "measured" if measured_calls else "not_applicable"
    expected_reasons: list[str] = []
    for call in calls:
        reason = call["reason"]
        if reason is not None and reason not in expected_reasons:
            expected_reasons.append(reason)
    expected_reason = None if measured_calls else "; ".join(expected_reasons)

    for field, expected in (
        ("model_call_count", len(calls)),
        ("measured_model_call_count", measured_calls),
        ("h2d_interval_count", expected_h2d_intervals),
        ("expert_compute_interval_count", expected_compute_intervals),
    ):
        if _integer(aggregate.get(field), f"{name}.{field}") != expected:
            raise ValueError(
                f"{name}.{field} is inconsistent with model-call timelines"
            )
    _timeline_number_equal(
        aggregate.get("h2d_union_ms"), expected_h2d_active, f"{name}.h2d_union_ms"
    )
    _timeline_number_equal(
        aggregate.get("expert_compute_union_ms"),
        expected_compute_active,
        f"{name}.expert_compute_union_ms",
    )
    _timeline_number_equal(
        aggregate.get("overlap_ms"), expected_overlap, f"{name}.overlap_ms"
    )
    _timeline_number_equal(
        aggregate.get("h2d_hidden_by_compute_ms"),
        expected_overlap,
        f"{name}.h2d_hidden_by_compute_ms",
    )
    _timeline_number_equal(
        aggregate.get("h2d_exposed_ms"),
        max(0.0, expected_h2d_active - expected_overlap),
        f"{name}.h2d_exposed_ms",
    )
    _timeline_number_equal(
        aggregate.get("active_duration_saved_by_overlap_ms"),
        expected_saved,
        f"{name}.active_duration_saved_by_overlap_ms",
    )
    _timeline_optional_ratio_equal(
        aggregate.get("h2d_overlap_fraction"),
        expected_overlap / expected_h2d_active if expected_h2d_active else None,
        f"{name}.h2d_overlap_fraction",
    )
    _timeline_optional_ratio_equal(
        aggregate.get("expert_compute_overlap_fraction"),
        (
            expected_overlap / expected_compute_active
            if expected_compute_active
            else None
        ),
        f"{name}.expert_compute_overlap_fraction",
    )
    if aggregate.get("status") != expected_status:
        raise ValueError(f"{name}.status is inconsistent with model-call timelines")
    if expected_status == "measured":
        if aggregate.get("reason") is not None:
            raise ValueError(f"{name}.reason must be null when status=measured")
    elif aggregate.get("reason") != expected_reason:
        raise ValueError(f"{name}.reason is inconsistent with model-call timelines")


def _validate_cuda_telemetry_pass(
    payload: dict[str, Any],
    mode: str,
    pass_name: str,
) -> None:
    passes = _mapping(payload.get("passes"), f"{mode}.passes")
    current = _mapping(passes.get(pass_name), f"{mode}.passes.{pass_name}")
    prefill = _mapping(current.get("prefill"), f"{mode}.{pass_name}.prefill")
    first = _mapping(current.get("first_token"), f"{mode}.{pass_name}.first_token")
    decode = _mapping(current.get("decode"), f"{mode}.{pass_name}.decode")
    per_token = decode.get("per_token")
    if not isinstance(per_token, list):
        raise ValueError(f"{mode}.{pass_name}.decode.per_token must be an array")

    prefill_call = _timeline_call(
        prefill.get("cuda_event_timeline"),
        f"{mode}.{pass_name}.prefill.cuda_event_timeline",
    )
    _timeline_call(
        first.get("cuda_event_timeline"),
        f"{mode}.{pass_name}.first_token.cuda_event_timeline",
    )
    if first.get("cuda_event_timeline") != prefill.get("cuda_event_timeline"):
        raise ValueError(f"{mode}.{pass_name}: first-token timeline must equal prefill")
    calls = [prefill_call]
    for index, token in enumerate(per_token):
        token_map = _mapping(token, f"{mode}.{pass_name}.decode.per_token[{index}]")
        calls.append(
            _timeline_call(
                token_map.get("cuda_event_timeline"),
                f"{mode}.{pass_name}.decode.per_token[{index}].cuda_event_timeline",
            )
        )
    _pass_cuda_overlap(
        current.get("cuda_overlap"),
        calls,
        f"{mode}.{pass_name}.cuda_overlap",
    )


def _validate_cuda_telemetry_absent_pass(
    payload: dict[str, Any],
    mode: str,
    pass_name: str,
) -> None:
    """Reject stale telemetry fields when a report did not request capture."""

    passes = _mapping(payload.get("passes"), f"{mode}.passes")
    current = _mapping(passes.get(pass_name), f"{mode}.passes.{pass_name}")
    prefill = _mapping(current.get("prefill"), f"{mode}.{pass_name}.prefill")
    first = _mapping(current.get("first_token"), f"{mode}.{pass_name}.first_token")
    decode = _mapping(current.get("decode"), f"{mode}.{pass_name}.decode")
    per_token = decode.get("per_token")
    if not isinstance(per_token, list):
        raise ValueError(f"{mode}.{pass_name}.decode.per_token must be an array")

    locations: list[tuple[dict[str, Any], str, str]] = [
        (prefill, "cuda_event_timeline", "prefill.cuda_event_timeline"),
        (first, "cuda_event_timeline", "first_token.cuda_event_timeline"),
        (current, "cuda_overlap", "cuda_overlap"),
    ]
    for index, raw_token in enumerate(per_token):
        token = _mapping(
            raw_token,
            f"{mode}.{pass_name}.decode.per_token[{index}]",
        )
        locations.append(
            (
                token,
                "cuda_event_timeline",
                f"decode.per_token[{index}].cuda_event_timeline",
            )
        )
    for container, field, location in locations:
        if field in container:
            raise ValueError(
                f"{mode}.{pass_name}.{location} must be absent when CUDA telemetry "
                "is not requested"
            )


def _sum_metrics(rows: list[dict[str, int]]) -> dict[str, int]:
    return {field: sum(row[field] for row in rows) for field in ADDITIVE_PRIMITIVES}


def _require_metric_sum(
    actual: dict[str, int], parts: list[dict[str, int]], name: str
) -> None:
    expected = _sum_metrics(parts)
    for field in ADDITIVE_PRIMITIVES:
        if actual[field] != expected[field]:
            raise ValueError(
                f"{name}.{field} is {actual[field]}, expected additive sum "
                f"{expected[field]}"
            )


def _pass_metric_scopes(
    payload: dict[str, Any], mode: str, pass_name: str, expert_bytes: int
) -> tuple[list[tuple[str, dict[str, int]]], dict[str, int]]:
    passes = _mapping(payload.get("passes"), f"{mode}.passes")
    current = _mapping(passes.get(pass_name), f"{mode}.passes.{pass_name}")
    prefill = _mapping(current.get("prefill"), f"{mode}.{pass_name}.prefill")
    first = _mapping(current.get("first_token"), f"{mode}.{pass_name}.first_token")
    decode = _mapping(current.get("decode"), f"{mode}.{pass_name}.decode")
    per_token = decode.get("per_token")
    if not isinstance(per_token, list):
        raise ValueError(f"{mode}.{pass_name}.decode.per_token must be an array")

    prefill_metrics = _metrics(
        prefill.get("metrics"), f"{mode}.{pass_name}.prefill.metrics", expert_bytes
    )
    first_metrics = _metrics(
        first.get("metrics"),
        f"{mode}.{pass_name}.first_token.metrics",
        expert_bytes,
    )
    if first_metrics != prefill_metrics:
        raise ValueError(f"{mode}.{pass_name}: first-token metrics must equal prefill")

    scopes: list[tuple[str, dict[str, int]]] = [
        ("prefill", prefill_metrics),
        ("first_token", first_metrics),
    ]
    additive_parts = [prefill_metrics]
    for index, item in enumerate(per_token):
        token = _mapping(item, f"{mode}.{pass_name}.decode.per_token[{index}]")
        token_metrics = _metrics(
            token.get("metrics"),
            f"{mode}.{pass_name}.decode.per_token[{index}].metrics",
            expert_bytes,
        )
        scopes.append((f"decode.per_token[{index}]", token_metrics))
        additive_parts.append(token_metrics)

    pass_metrics = _metrics(
        current.get("metrics"), f"{mode}.{pass_name}.metrics", expert_bytes
    )
    _require_metric_sum(pass_metrics, additive_parts, f"{mode}.{pass_name}.metrics")
    scopes.append(("metrics", pass_metrics))
    return scopes, pass_metrics


def _validate_qwen_correctness_smoke(
    payload: dict[str, Any],
    mode: str,
    evidence: dict[str, Any],
) -> None:
    """Require the narrow, non-publishable full-checkpoint acceptance contract."""

    if evidence.get("publishable_benchmark_evidence") is not False:
        raise ValueError(
            f"{mode}: correctness-smoke input must be explicitly non-publishable"
        )
    if evidence.get("offline_local_only") is not True:
        raise ValueError(f"{mode}: correctness smoke must be offline-local-only")

    model = _mapping(payload.get("model"), f"{mode}.model")
    if model.get("checkpoint_profile") != "qwen2-moe":
        raise ValueError(
            f"{mode}: correctness-smoke comparison only supports qwen2-moe"
        )
    if model.get("verification_scope") != "full_required_file_manifest":
        raise ValueError(
            f"{mode}: qwen2-moe correctness smoke requires full-manifest verification"
        )
    if model.get("preflight_full_manifest_verified") is not True:
        raise ValueError(
            f"{mode}: qwen2-moe correctness smoke requires pre-use manifest verification"
        )
    verified_files = _mapping(model.get("verified_files"), f"{mode}.verified_files")
    if not verified_files:
        raise ValueError(f"{mode}: verified_files cannot be empty")

    reference = _mapping(
        payload.get("reference_comparison"), f"{mode}.reference_comparison"
    )
    if (
        reference.get("available") is not True
        or reference.get("matched") is not True
        or reference.get("mode") != "autoregressive_exact_gate"
        or reference.get("first_mismatch_index") is not None
    ):
        raise ValueError(
            f"{mode}: correctness smoke requires an exact autoregressive reference match"
        )
    reference_ids = reference.get("generated_token_ids")
    if not isinstance(reference_ids, list) or not reference_ids:
        raise ValueError(f"{mode}: reference token ids must be a non-empty array")
    if any(
        isinstance(token, bool) or not isinstance(token, int) for token in reference_ids
    ):
        raise ValueError(f"{mode}: reference token ids must contain integers")
    reference_count = len(reference_ids)
    for field in ("matched_tokens", "total_tokens", "source_generated_token_count"):
        if (
            _integer(reference.get(field), f"{mode}.reference.{field}")
            != reference_count
        ):
            raise ValueError(
                f"{mode}: reference.{field} must cover every generated token"
            )
    if _number(reference.get("temperature"), f"{mode}.reference.temperature") != 0.0:
        raise ValueError(f"{mode}: correctness smoke requires greedy temperature 0")

    passes = _mapping(payload.get("passes"), f"{mode}.passes")
    for pass_name in PASS_NAMES:
        current = _mapping(passes.get(pass_name), f"{mode}.{pass_name}")
        if current.get("teacher_forced") is not False:
            raise ValueError(
                f"{mode}.{pass_name}: correctness smoke must be autoregressive"
            )
        if current.get("generated_token_count") != reference_count:
            raise ValueError(
                f"{mode}.{pass_name}: generated-token count must match the reference"
            )
        if current.get("generated_ids") != reference_ids:
            raise ValueError(
                f"{mode}.{pass_name}: generated ids must exactly match the reference"
            )
        if current.get("fed_token_ids") != reference_ids:
            raise ValueError(
                f"{mode}.{pass_name}: fed ids must be the autoregressive outputs"
            )


def _validate_mode(
    payload: dict[str, Any],
    mode: str,
    *,
    evidence_scope: str = "benchmark",
) -> dict[str, Any]:
    if payload.get("status") != "ok" or payload.get("schema_version") != 1:
        raise ValueError(f"{mode}: expected status=ok and schema_version=1")
    evidence = _mapping(payload.get("evidence"), f"{mode}.evidence")
    publishable = evidence.get("publishable_benchmark_evidence")
    if evidence_scope == "benchmark":
        if publishable is not None and publishable is not True:
            raise ValueError(f"{mode}: demo output is not benchmark evidence")
    elif evidence_scope == "correctness-smoke":
        _validate_qwen_correctness_smoke(payload, mode, evidence)
    else:
        raise ValueError(f"unsupported evidence scope: {evidence_scope}")
    source = _mapping(payload.get("source"), f"{mode}.source")
    if source.get("provenance_mode") == "demo":
        raise ValueError(f"{mode}: demo provenance is not benchmark evidence")
    if source.get("tree_clean") is not True:
        raise ValueError(f"{mode}: benchmark source must be a clean Git tree")
    commit = source.get("commit")
    script_hash = source.get("benchmark_script_sha256")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{mode}: source.commit must be a full Git commit")
    if not isinstance(script_hash, str) or len(script_hash) != 64:
        raise ValueError(f"{mode}: benchmark script SHA-256 is missing")

    runtime = _mapping(payload.get("runtime"), f"{mode}.runtime")
    if runtime.get("pipeline") != mode:
        raise ValueError(f"{mode}: runtime.pipeline must equal {mode!r}")
    telemetry_requested = _cuda_overlap_requested(runtime, mode)
    budget = _mapping(runtime.get("budget"), f"{mode}.runtime.budget")
    if budget.get("pipeline") != mode:
        raise ValueError(f"{mode}: runtime.budget.pipeline must equal {mode!r}")
    expert_bytes = _integer(budget.get("expert_bytes"), f"{mode}.expert_bytes")
    if expert_bytes == 0:
        raise ValueError(f"{mode}: expert_bytes must be positive")

    model_load = _mapping(payload.get("model_load"), f"{mode}.model_load")
    preload = _metrics(
        model_load.get("static_preload_metrics"),
        f"{mode}.model_load.static_preload_metrics",
        expert_bytes,
    )
    pass_results: dict[str, dict[str, Any]] = {}
    final_parts = [preload]
    for pass_name in PASS_NAMES:
        scopes, pass_metrics = _pass_metric_scopes(
            payload, mode, pass_name, expert_bytes
        )
        current = _mapping(
            _mapping(payload["passes"], f"{mode}.passes").get(pass_name),
            f"{mode}.{pass_name}",
        )
        wall = _number(
            current.get("total_wall_seconds"),
            f"{mode}.{pass_name}.total_wall_seconds",
            positive=True,
        )
        if telemetry_requested:
            _validate_cuda_telemetry_pass(payload, mode, pass_name)
        else:
            _validate_cuda_telemetry_absent_pass(payload, mode, pass_name)
        pass_results[pass_name] = {
            "wall_seconds": wall,
            "scopes": scopes,
            "metrics": pass_metrics,
        }
        final_parts.append(pass_metrics)

    final_metrics = _metrics(
        runtime.get("final_metrics"), f"{mode}.runtime.final_metrics", expert_bytes
    )
    _require_metric_sum(final_metrics, final_parts, f"{mode}.runtime.final_metrics")

    staging_slots = _integer(budget.get("staging_slots"), f"{mode}.staging_slots")
    peak_staging = _integer(
        _mapping(runtime.get("final_metrics"), f"{mode}.final_metrics").get(
            "peak_staging_in_use"
        ),
        f"{mode}.peak_staging_in_use",
    )
    if peak_staging > staging_slots:
        raise ValueError(f"{mode}: peak staging use exceeds the staging budget")
    pending_peak = _integer(
        _mapping(runtime.get("final_metrics"), f"{mode}.final_metrics").get(
            "pending_loads_peak"
        ),
        f"{mode}.pending_loads_peak",
    )
    if pending_peak > 2 * staging_slots:
        raise ValueError(
            f"{mode}: pending-load peak exceeds the two-window benchmark-forward bound"
        )
    return {
        "source": source,
        "runtime": runtime,
        "budget": budget,
        "expert_bytes": expert_bytes,
        "preload": preload,
        "passes": pass_results,
        "final_metrics": final_metrics,
    }


def _compare_identity(sync: dict[str, Any], async_: dict[str, Any]) -> None:
    _require_equal(sync.get("source"), async_.get("source"), "source")
    _require_equal(sync.get("environment"), async_.get("environment"), "environment")
    _require_equal(sync.get("workload"), async_.get("workload"), "workload")

    sync_model = dict(_mapping(sync.get("model"), "sync.model"))
    async_model = dict(_mapping(async_.get("model"), "async.model"))
    sync_model.pop("hash_verification_seconds", None)
    async_model.pop("hash_verification_seconds", None)
    _require_equal(sync_model, async_model, "model identity")

    sync_runtime = _mapping(sync.get("runtime"), "sync.runtime")
    async_runtime = _mapping(async_.get("runtime"), "async.runtime")
    for field in RUNTIME_IDENTITY_FIELDS:
        _require_equal(
            sync_runtime.get(field), async_runtime.get(field), f"runtime.{field}"
        )
    sync_budget = dict(_mapping(sync_runtime.get("budget"), "sync.runtime.budget"))
    async_budget = dict(_mapping(async_runtime.get("budget"), "async.runtime.budget"))
    sync_budget.pop("pipeline", None)
    async_budget.pop("pipeline", None)
    # This records the scheduling path resolved for each pass.  It is expected
    # to differ between the sync and async arms, just like ``pipeline`` above;
    # it is not a hardware/cache-budget identity field.
    sync_budget.pop("resolved_pipeline_by_pass", None)
    async_budget.pop("resolved_pipeline_by_pass", None)
    _require_equal(sync_budget, async_budget, "runtime.budget")

    for pass_name in PASS_NAMES:
        sync_pass = _mapping(
            _mapping(sync.get("passes"), "sync.passes").get(pass_name),
            f"sync.{pass_name}",
        )
        async_pass = _mapping(
            _mapping(async_.get("passes"), "async.passes").get(pass_name),
            f"async.{pass_name}",
        )
        for field in PASS_IDENTITY_FIELDS:
            _require_equal(
                sync_pass.get(field),
                async_pass.get(field),
                f"passes.{pass_name}.{field}",
            )
        for scope in ("prefill", "first_token"):
            left = _mapping(sync_pass.get(scope), f"sync.{pass_name}.{scope}")
            right = _mapping(async_pass.get(scope), f"async.{pass_name}.{scope}")
            for field in (
                "input_tokens",
                "index",
                "fed_token_id",
                "token_id",
                "source",
            ):
                if field in left or field in right:
                    _require_equal(
                        left.get(field),
                        right.get(field),
                        f"{pass_name}.{scope}.{field}",
                    )
        sync_tokens = _mapping(sync_pass.get("decode"), "sync.decode").get("per_token")
        async_tokens = _mapping(async_pass.get("decode"), "async.decode").get(
            "per_token"
        )
        if not isinstance(sync_tokens, list) or not isinstance(async_tokens, list):
            raise ValueError(f"{pass_name}: decode.per_token must be arrays")
        _require_equal(len(sync_tokens), len(async_tokens), f"{pass_name}.decode count")
        for index, (left, right) in enumerate(
            zip(sync_tokens, async_tokens, strict=True)
        ):
            left_map = _mapping(left, f"sync.{pass_name}.decode[{index}]")
            right_map = _mapping(right, f"async.{pass_name}.decode[{index}]")
            for field in (
                "index",
                "predicted_token_id",
                "fed_token_id",
                "token_id",
                "matched_forced_token",
                "source",
            ):
                _require_equal(
                    left_map.get(field),
                    right_map.get(field),
                    f"{pass_name}.decode[{index}].{field}",
                )

    sync_reference = _mapping(sync.get("reference_comparison"), "sync.reference")
    async_reference = _mapping(async_.get("reference_comparison"), "async.reference")
    for field in REFERENCE_IDENTITY_FIELDS:
        _require_equal(
            sync_reference.get(field),
            async_reference.get(field),
            f"reference_comparison.{field}",
        )


def compare_reports(
    sync: dict[str, Any],
    async_: dict[str, Any],
    *,
    evidence_scope: str = "benchmark",
) -> dict[str, Any]:
    """Validate two already-loaded reports and return a paired summary."""
    _compare_identity(sync, async_)
    validated_sync = _validate_mode(sync, "sync", evidence_scope=evidence_scope)
    validated_async = _validate_mode(async_, "async", evidence_scope=evidence_scope)
    _require_equal(
        validated_sync["expert_bytes"],
        validated_async["expert_bytes"],
        "expert_bytes",
    )
    _require_equal(validated_sync["preload"], validated_async["preload"], "preload")
    _require_equal(
        validated_sync["final_metrics"],
        validated_async["final_metrics"],
        "runtime.final_metrics",
    )

    comparisons: dict[str, Any] = {}
    for pass_name in PASS_NAMES:
        sync_pass = validated_sync["passes"][pass_name]
        async_pass = validated_async["passes"][pass_name]
        _require_equal(
            sync_pass["scopes"], async_pass["scopes"], f"{pass_name}.metric scopes"
        )
        sync_memory = _mapping(
            _mapping(sync["passes"], "sync.passes")[pass_name].get("cuda_memory"),
            f"sync.{pass_name}.cuda_memory",
        )
        async_memory = _mapping(
            _mapping(async_["passes"], "async.passes")[pass_name].get("cuda_memory"),
            f"async.{pass_name}.cuda_memory",
        )
        sync_peak = _integer(
            sync_memory.get("peak_allocated_bytes"),
            f"sync.{pass_name}.peak_allocated_bytes",
        )
        async_peak = _integer(
            async_memory.get("peak_allocated_bytes"),
            f"async.{pass_name}.peak_allocated_bytes",
        )
        if async_peak > sync_peak + MAX_EXTRA_ASYNC_PEAK_VRAM_BYTES:
            raise ValueError(
                f"{pass_name}: async peak VRAM exceeds the 64 MiB comparison guard"
            )
        sync_wall = sync_pass["wall_seconds"]
        async_wall = async_pass["wall_seconds"]
        comparisons[pass_name] = {
            "sync_wall_seconds": sync_wall,
            "async_wall_seconds": async_wall,
            "sync_over_async_ratio": sync_wall / async_wall,
            "saving_seconds": sync_wall - async_wall,
            "saving_fraction": (sync_wall - async_wall) / sync_wall,
            "sync_peak_allocated_vram_bytes": sync_peak,
            "async_peak_allocated_vram_bytes": async_peak,
            "metrics": sync_pass["metrics"],
        }

    return {
        "schema_version": 1,
        "status": "ok",
        "source_commit": validated_sync["source"]["commit"],
        "benchmark_script_sha256": validated_sync["source"]["benchmark_script_sha256"],
        "exact_invariants": True,
        "evidence_scope": evidence_scope,
        "publishable_benchmark_evidence": evidence_scope == "benchmark",
        "passes": comparisons,
        "limitations": [
            (
                "This validates one non-publishable Qwen correctness pair, not benchmark evidence."
                if evidence_scope == "correctness-smoke"
                else "This validates one paired two-token smoke, not a general speedup claim."
            ),
            "Timing can still vary with OS page-cache, clocks, and background load.",
            "Counter equality does not itself prove physical NVMe or CUDA interval overlap.",
        ],
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare_pair(
    sync_path: Path,
    async_path: Path,
    *,
    evidence_scope: str = "benchmark",
) -> dict[str, Any]:
    report = compare_reports(
        _load_json(sync_path),
        _load_json(async_path),
        evidence_scope=evidence_scope,
    )
    report["inputs"] = {
        "sync": {
            "path": str(sync_path.resolve()),
            "sha256": _sha256(sync_path),
        },
        "async": {
            "path": str(async_path.resolve()),
            "sha256": _sha256(async_path),
        },
    }
    return report


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sync", type=Path, help="sync result JSON")
    parser.add_argument("async_path", type=Path, help="async result JSON")
    parser.add_argument("--output", type=Path, help="optional create-only report JSON")
    parser.add_argument(
        "--correctness-smoke",
        action="store_true",
        help=(
            "accept only a non-publishable, full-manifest Qwen pair with an "
            "exact autoregressive reference match"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = compare_pair(
            args.sync,
            args.async_path,
            evidence_scope=(
                "correctness-smoke" if args.correctness_smoke else "benchmark"
            ),
        )
        if args.output is not None:
            _write_create_only(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except ValueError as exc:
        print(f"comparison failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
