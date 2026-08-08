from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .simulator import ComparisonResult, RunMetrics


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _run_lines(metrics: RunMetrics) -> list[str]:
    return [
        f"Mode:                 {metrics.mode}",
        f"Estimated tokens/s:   {metrics.tokens_per_second:.3f}",
        f"Estimated elapsed:    {metrics.elapsed_ms / 1000.0:.3f} s",
        f"Compute time:         {metrics.compute_ms / 1000.0:.3f} s",
        f"Demand stall:         {metrics.demand_stall_ms / 1000.0:.3f} s",
        f"Prefetch stall:       {metrics.prefetch_stall_ms / 1000.0:.3f} s",
        f"VRAM demand hit-rate: {format_percent(metrics.demand_l1_hit_rate)}",
        f"RAM+VRAM hit-rate:    {format_percent(metrics.demand_cache_hit_rate)}",
        f"Demand NVMe traffic:  {format_bytes(metrics.demand_nvme_to_ram_bytes)}",
        f"Total NVMe traffic:   {format_bytes(metrics.total_nvme_to_ram_bytes)}",
        f"Total RAM→VRAM:       {format_bytes(metrics.total_ram_to_vram_bytes)}",
        f"Prefetch precision:   {format_percent(metrics.prefetch_precision)}",
        f"Predicted/deadline-admitted: {metrics.prefetch_predictions:,}/{metrics.prefetch_candidates:,}",
        f"Deadline rejections:  {metrics.prefetch_rejected_deadline:,}",
        f"Capacity rejections:  {metrics.prefetch_rejected_capacity:,}",
    ]


def comparison_console(result: ComparisonResult, config: ExperimentConfig) -> str:
    lines = [
        "MoEVM Lab — simulation report",
        "=" * 31,
        "SIMULATION ONLY: these values are not measured model-generation performance.",
        f"Model shape: {config.model.name}",
        (
            f"Tokens: {result.baseline.tokens}; layers: {config.model.layers}; "
            f"experts/layer: {config.model.experts_per_layer}; top-k: {config.model.top_k}"
        ),
        "",
        "Baseline",
        "--------",
        *_run_lines(result.baseline),
        "",
        "Predictive prefetch",
        "-------------------",
        *_run_lines(result.prefetch),
        "",
        "Comparison",
        "----------",
        f"Estimated speedup:          {result.speedup:.3f}x",
        f"Demand NVMe reduction:      {format_percent(result.demand_nvme_reduction)}",
        f"Total NVMe reduction:       {format_percent(result.total_nvme_reduction)}",
        f"Demand stall reduction:     {format_percent(result.demand_stall_reduction)}",
        f"RAM→VRAM traffic change:    {format_percent(result.ram_to_vram_traffic_change)}",
    ]
    return "\n".join(lines)


def comparison_markdown(result: ComparisonResult, config: ExperimentConfig) -> str:
    baseline = result.baseline
    prefetch = result.prefetch
    return f"""# MoEVM Lab simulation result

> **Simulation only.** This report does not claim measured LLM token-generation performance.

## Experiment

| Field | Value |
|---|---:|
| Model shape | `{config.model.name}` |
| Tokens | {baseline.tokens} |
| Layers | {config.model.layers} |
| Experts per layer | {config.model.experts_per_layer} |
| Selected experts | {config.model.top_k} |
| Expert size | {format_bytes(config.model.expert_size_bytes)} |
| VRAM expert cache | {format_bytes(config.hardware.vram_cache_bytes)} |
| RAM expert cache | {format_bytes(config.hardware.ram_cache_bytes)} |

## Results

| Metric | Baseline | Predictive prefetch |
|---|---:|---:|
| Estimated tokens/s | {baseline.tokens_per_second:.3f} | {prefetch.tokens_per_second:.3f} |
| Estimated elapsed | {baseline.elapsed_ms / 1000.0:.3f} s | {prefetch.elapsed_ms / 1000.0:.3f} s |
| Demand stall | {baseline.demand_stall_ms / 1000.0:.3f} s | {prefetch.demand_stall_ms / 1000.0:.3f} s |
| Prefetch stall | {baseline.prefetch_stall_ms / 1000.0:.3f} s | {prefetch.prefetch_stall_ms / 1000.0:.3f} s |
| VRAM demand hit-rate | {format_percent(baseline.demand_l1_hit_rate)} | {format_percent(prefetch.demand_l1_hit_rate)} |
| RAM+VRAM demand hit-rate | {format_percent(baseline.demand_cache_hit_rate)} | {format_percent(prefetch.demand_cache_hit_rate)} |
| Demand NVMe traffic | {format_bytes(baseline.demand_nvme_to_ram_bytes)} | {format_bytes(prefetch.demand_nvme_to_ram_bytes)} |
| Total NVMe traffic | {format_bytes(baseline.total_nvme_to_ram_bytes)} | {format_bytes(prefetch.total_nvme_to_ram_bytes)} |
| Total RAM→VRAM traffic | {format_bytes(baseline.total_ram_to_vram_bytes)} | {format_bytes(prefetch.total_ram_to_vram_bytes)} |
| Prefetch precision | — | {format_percent(prefetch.prefetch_precision)} |
| Predicted/deadline-admitted prefetches | — | {prefetch.prefetch_predictions:,} / {prefetch.prefetch_candidates:,} |
| Deadline rejections | — | {prefetch.prefetch_rejected_deadline:,} |
| Capacity rejections | — | {prefetch.prefetch_rejected_capacity:,} |

## Comparison

- Estimated speedup: **{result.speedup:.3f}x**
- Demand-path NVMe reduction: **{format_percent(result.demand_nvme_reduction)}**
- Total NVMe reduction: **{format_percent(result.total_nvme_reduction)}**
- Demand-stall reduction: **{format_percent(result.demand_stall_reduction)}**
- RAM→VRAM traffic change: **{format_percent(result.ram_to_vram_traffic_change)}**

A useful prefetch can reduce blocking demand reads while still increasing total traffic. Both latency and traffic figures must therefore be reported.
"""


def write_comparison(
    output_dir: str | Path,
    result: ComparisonResult,
    config: ExperimentConfig,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    markdown_path = destination / "comparison.md"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": config.as_dict(),
        **result.to_dict(),
    }
    json_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(comparison_markdown(result, config), encoding="utf-8")
    return json_path, markdown_path
