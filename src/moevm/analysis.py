from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from .config import PredictorConfig
from .predictor import OnlineExpertPredictor
from .types import RoutingStep


@dataclass(frozen=True, slots=True)
class LayerTraceStats:
    layer_index: int
    tokens: int
    unique_experts: int
    normalized_entropy: float
    temporal_overlap: float
    exact_repeat_rate: float
    mean_topk_probability_mass: float | None
    top_experts: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class PredictorTraceStats:
    evaluated_steps: int
    steps_with_predictions: int
    predicted_experts: int
    correct_experts: int
    precision: float
    recall: float
    coverage: float


@dataclass(frozen=True, slots=True)
class RoutingTraceAnalysis:
    token_start: int
    tokens: int
    layers: int
    top_k: int
    steps: int
    expert_accesses: int
    score_coverage: float
    mean_topk_probability_mass: float | None
    mean_temporal_overlap: float
    exact_repeat_rate: float
    mean_normalized_entropy: float
    predictor: PredictorTraceStats | None
    per_layer: tuple[LayerTraceStats, ...]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _validate_complete_grid(
    steps: list[RoutingStep],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if not steps:
        raise ValueError("trace cannot be empty")

    tokens = tuple(sorted({step.token_index for step in steps}))
    layers = tuple(sorted({step.layer_index for step in steps}))
    top_k = len(steps[0].experts)
    if tokens != tuple(range(tokens[0], tokens[0] + len(tokens))):
        raise ValueError("trace token indices must be contiguous")
    if layers != tuple(range(len(layers))):
        raise ValueError("trace layer indices must start at zero and be contiguous")
    if any(len(step.experts) != top_k for step in steps):
        raise ValueError("every trace step must select the same number of experts")

    addresses = {(step.token_index, step.layer_index) for step in steps}
    if len(addresses) != len(steps):
        raise ValueError("trace contains duplicate token/layer steps")
    expected = {(token, layer) for token in tokens for layer in layers}
    if addresses != expected:
        raise ValueError("trace must contain one step for every token/layer pair")
    return tokens, layers, top_k


def _normalized_entropy(counts: Counter[int], experts_per_layer: int) -> float:
    total = sum(counts.values())
    if total == 0 or experts_per_layer <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return entropy / math.log(experts_per_layer)


def _mean_probability_mass(steps: list[RoutingStep]) -> float | None:
    masses = [sum(step.scores) for step in steps if step.scores is not None]
    if not masses:
        return None
    return sum(masses) / len(masses)


def _evaluate_predictor(
    steps: list[RoutingStep],
    config: PredictorConfig,
    top_k: int,
    first_token: int,
) -> PredictorTraceStats:
    predictor = OnlineExpertPredictor(config)
    previous_step: RoutingStep | None = None
    evaluated_steps = 0
    steps_with_predictions = 0
    predicted_experts = 0
    correct_experts = 0
    target_experts = 0

    for step in steps:
        current_step = (
            previous_step
            if previous_step is not None
            and previous_step.token_index == step.token_index
            and previous_step.layer_index + 1 == step.layer_index
            else None
        )
        prediction = predictor.predict(
            target_layer=step.layer_index,
            current_step=current_step,
            limit=top_k,
        )
        if step.token_index > first_token:
            evaluated_steps += 1
            target_experts += len(step.experts)
            predicted_experts += len(prediction)
            if prediction:
                steps_with_predictions += 1
            correct_experts += len(set(prediction).intersection(step.experts))
        predictor.observe(step, previous_step)
        previous_step = step

    return PredictorTraceStats(
        evaluated_steps=evaluated_steps,
        steps_with_predictions=steps_with_predictions,
        predicted_experts=predicted_experts,
        correct_experts=correct_experts,
        precision=(
            0.0 if predicted_experts == 0 else correct_experts / predicted_experts
        ),
        recall=0.0 if target_experts == 0 else correct_experts / target_experts,
        coverage=(
            0.0 if evaluated_steps == 0 else steps_with_predictions / evaluated_steps
        ),
    )


def analyze_routing_trace(
    steps: list[RoutingStep],
    *,
    experts_per_layer: int,
    predictor_config: PredictorConfig | None = None,
) -> RoutingTraceAnalysis:
    """Measure locality, entropy, router confidence, and online predictability."""
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")
    tokens, layers, top_k = _validate_complete_grid(steps)
    ordered = sorted(steps, key=lambda step: (step.token_index, step.layer_index))
    if any(max(step.experts) >= experts_per_layer for step in ordered):
        raise ValueError("trace expert id exceeds experts_per_layer")

    per_layer: list[LayerTraceStats] = []
    overlap_sum = 0.0
    repeat_count = 0
    temporal_pairs = 0
    entropy_sum = 0.0

    for layer in layers:
        layer_steps = [step for step in ordered if step.layer_index == layer]
        counts: Counter[int] = Counter()
        for step in layer_steps:
            counts.update(step.experts)

        layer_overlap_sum = 0.0
        layer_repeats = 0
        for previous, current in pairwise(layer_steps):
            overlap = len(set(previous.experts).intersection(current.experts)) / top_k
            layer_overlap_sum += overlap
            layer_repeats += previous.experts == current.experts

        layer_pairs = max(0, len(layer_steps) - 1)
        entropy = _normalized_entropy(counts, experts_per_layer)
        total_accesses = sum(counts.values())
        top_experts = tuple(
            (expert, count / total_accesses)
            for expert, count in counts.most_common(min(8, len(counts)))
        )
        per_layer.append(
            LayerTraceStats(
                layer_index=layer,
                tokens=len(layer_steps),
                unique_experts=len(counts),
                normalized_entropy=entropy,
                temporal_overlap=(
                    0.0 if layer_pairs == 0 else layer_overlap_sum / layer_pairs
                ),
                exact_repeat_rate=(
                    0.0 if layer_pairs == 0 else layer_repeats / layer_pairs
                ),
                mean_topk_probability_mass=_mean_probability_mass(layer_steps),
                top_experts=top_experts,
            )
        )
        overlap_sum += layer_overlap_sum
        repeat_count += layer_repeats
        temporal_pairs += layer_pairs
        entropy_sum += entropy

    scored_steps = sum(step.scores is not None for step in ordered)
    predictor = (
        _evaluate_predictor(ordered, predictor_config, top_k, tokens[0])
        if predictor_config is not None and predictor_config.enabled
        else None
    )
    return RoutingTraceAnalysis(
        token_start=tokens[0],
        tokens=len(tokens),
        layers=len(layers),
        top_k=top_k,
        steps=len(ordered),
        expert_accesses=len(ordered) * top_k,
        score_coverage=scored_steps / len(ordered),
        mean_topk_probability_mass=_mean_probability_mass(ordered),
        mean_temporal_overlap=(
            0.0 if temporal_pairs == 0 else overlap_sum / temporal_pairs
        ),
        exact_repeat_rate=(
            0.0 if temporal_pairs == 0 else repeat_count / temporal_pairs
        ),
        mean_normalized_entropy=entropy_sum / len(layers),
        predictor=predictor,
        per_layer=tuple(per_layer),
    )


def _percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def trace_analysis_console(analysis: RoutingTraceAnalysis) -> str:
    lines = [
        "MoEVM Lab — routing trace analysis",
        "==================================",
        "TRACE REPLAY: routing evidence only; no transfer timing is measured here.",
        f"Tokens/layers/top-k: {analysis.tokens}/{analysis.layers}/{analysis.top_k}",
        f"Expert accesses: {analysis.expert_accesses:,}",
        f"Router-score coverage: {_percent(analysis.score_coverage)}",
        f"Mean temporal overlap: {_percent(analysis.mean_temporal_overlap)}",
        f"Exact repeat rate: {_percent(analysis.exact_repeat_rate)}",
        f"Mean normalized entropy: {_percent(analysis.mean_normalized_entropy)}",
    ]
    if analysis.mean_topk_probability_mass is not None:
        lines.append(
            "Mean selected probability mass: "
            f"{_percent(analysis.mean_topk_probability_mass)}"
        )
    if analysis.predictor is not None:
        lines.extend(
            [
                "",
                "Online predictor after first-token warm-up",
                "------------------------------------------",
                f"Coverage:  {_percent(analysis.predictor.coverage)}",
                f"Precision: {_percent(analysis.predictor.precision)}",
                f"Recall:    {_percent(analysis.predictor.recall)}",
            ]
        )
    return "\n".join(lines)


def trace_analysis_markdown(analysis: RoutingTraceAnalysis) -> str:
    predictor_rows = ""
    if analysis.predictor is not None:
        predictor_rows = f"""
| Predictor coverage | {_percent(analysis.predictor.coverage)} |
| Predictor precision | {_percent(analysis.predictor.precision)} |
| Predictor recall | {_percent(analysis.predictor.recall)} |"""
    probability_row = ""
    if analysis.mean_topk_probability_mass is not None:
        probability_row = (
            "\n| Mean selected probability mass | "
            f"{_percent(analysis.mean_topk_probability_mass)} |"
        )
    layer_rows = "\n".join(
        f"| {layer.layer_index} | {layer.unique_experts} | "
        f"{_percent(layer.normalized_entropy)} | "
        f"{_percent(layer.temporal_overlap)} | "
        f"{_percent(layer.exact_repeat_rate)} |"
        for layer in analysis.per_layer
    )
    return f"""# MoEVM Lab routing trace analysis

> **TRACE REPLAY:** routing evidence only; no transfer timing is measured here.

| Metric | Value |
|---|---:|
| Tokens | {analysis.tokens:,} |
| Layers | {analysis.layers:,} |
| Top-k | {analysis.top_k:,} |
| Expert accesses | {analysis.expert_accesses:,} |
| Router-score coverage | {_percent(analysis.score_coverage)} |
| Mean temporal overlap | {_percent(analysis.mean_temporal_overlap)} |
| Exact repeat rate | {_percent(analysis.exact_repeat_rate)} |
| Mean normalized entropy | {_percent(analysis.mean_normalized_entropy)} |{probability_row}{predictor_rows}

## Per-layer routing

| Layer | Unique experts | Normalized entropy | Temporal overlap | Exact repeats |
|---:|---:|---:|---:|---:|
{layer_rows}
"""


def write_trace_analysis(
    output_dir: str | Path,
    analysis: RoutingTraceAnalysis,
    *,
    trace_path: str | Path | None = None,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "label": "routing analysis",
        "analysis": analysis.to_dict(),
    }
    if trace_path is not None:
        source = Path(trace_path)
        payload["trace"] = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    json_path = destination / "routing_analysis.json"
    markdown_path = destination / "routing_analysis.md"
    json_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(trace_analysis_markdown(analysis), encoding="utf-8")
    return json_path, markdown_path
