from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class ExpertKey:
    """Logical expert address inside a model."""

    layer: int
    expert: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.expert < 0:
            raise ValueError("expert must be non-negative")

    def compact(self) -> str:
        return f"L{self.layer}:E{self.expert}"


@dataclass(frozen=True, slots=True)
class RoutingStep:
    """Experts selected for one token at one MoE layer."""

    token_index: int
    layer_index: int
    experts: tuple[int, ...]
    scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.token_index < 0:
            raise ValueError("token_index must be non-negative")
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if not self.experts:
            raise ValueError("experts cannot be empty")
        if len(set(self.experts)) != len(self.experts):
            raise ValueError("experts must be unique within a routing step")
        if min(self.experts) < 0:
            raise ValueError("expert ids must be non-negative")
        if self.scores is not None:
            if len(self.scores) != len(self.experts):
                raise ValueError("router scores must align with selected experts")
            if any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
                for score in self.scores
            ):
                raise ValueError("router scores must be finite probabilities in [0, 1]")

    @property
    def keys(self) -> tuple[ExpertKey, ...]:
        return tuple(ExpertKey(self.layer_index, expert) for expert in self.experts)
