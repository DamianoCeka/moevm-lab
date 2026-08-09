from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import ExperimentConfig
from .types import RoutingStep


def _json_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _json_probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field} must be a finite probability in [0, 1]")
    return converted


class SyntheticRoutingTrace:
    """Generate domain-local, temporally correlated MoE routing traces."""

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate()
        self.config = config
        self._random = random.Random(config.trace.seed)
        hotset_size = min(
            config.model.experts_per_layer,
            max(
                config.model.top_k,
                int(config.model.top_k * config.trace.hotset_multiplier),
            ),
        )
        self._hotsets: list[list[tuple[int, ...]]] = []
        for _domain in range(config.trace.domains):
            layers: list[tuple[int, ...]] = []
            for _layer in range(config.model.layers):
                layers.append(
                    tuple(
                        self._random.sample(
                            range(config.model.experts_per_layer),
                            k=hotset_size,
                        )
                    )
                )
            self._hotsets.append(layers)

    def _choose_experts(
        self,
        domain: int,
        layer: int,
        previous: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        model = self.config.model
        trace = self.config.trace
        selected: list[int] = []

        if previous:
            retained = list(previous)
            self._random.shuffle(retained)
            for expert in retained:
                if (
                    len(selected) < model.top_k
                    and self._random.random() < trace.temporal_reuse_probability
                ):
                    selected.append(expert)

        preferred = list(self._hotsets[domain][layer])
        self._random.shuffle(preferred)
        while len(selected) < model.top_k:
            use_random = self._random.random() < trace.random_expert_probability
            if use_random:
                candidate = self._random.randrange(model.experts_per_layer)
            elif preferred:
                candidate = preferred.pop()
            else:
                candidate = self._random.randrange(model.experts_per_layer)
            if candidate not in selected:
                selected.append(candidate)

        return tuple(sorted(selected))

    def generate(self) -> Iterator[RoutingStep]:
        domain = self._random.randrange(self.config.trace.domains)
        previous_by_layer: dict[int, tuple[int, ...]] = {}

        for token in range(self.config.trace.tokens):
            if (
                token
                and self._random.random() < self.config.trace.domain_switch_probability
            ):
                choices = [d for d in range(self.config.trace.domains) if d != domain]
                if choices:
                    domain = self._random.choice(choices)

            for layer in range(self.config.model.layers):
                experts = self._choose_experts(
                    domain=domain,
                    layer=layer,
                    previous=previous_by_layer.get(layer),
                )
                previous_by_layer[layer] = experts
                yield RoutingStep(token, layer, experts)


def write_trace(path: str | Path, steps: Iterable[RoutingStep]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for step in steps:
            row: dict[str, object] = {
                "token": step.token_index,
                "layer": step.layer_index,
                "experts": list(step.experts),
            }
            if step.scores is not None:
                row["scores"] = list(step.scores)
            handle.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                )
                + "\n"
            )


def read_trace(path: str | Path) -> list[RoutingStep]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"trace file not found: {source}")

    steps: list[RoutingStep] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each trace row must be a JSON object")
                experts = row["experts"]
                if not isinstance(experts, list):
                    raise ValueError("experts must be a JSON array")
                raw_scores = row.get("scores")
                if raw_scores is not None and not isinstance(raw_scores, list):
                    raise ValueError("scores must be a JSON array")
                steps.append(
                    RoutingStep(
                        token_index=_json_integer(row["token"], "token"),
                        layer_index=_json_integer(row["layer"], "layer"),
                        experts=tuple(
                            _json_integer(value, "expert") for value in experts
                        ),
                        scores=(
                            tuple(
                                _json_probability(value, "score")
                                for value in raw_scores
                            )
                            if raw_scores is not None
                            else None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid trace at line {line_number}: {exc}") from exc
    if not steps:
        raise ValueError("trace is empty")
    return steps
