from __future__ import annotations

from collections import Counter, defaultdict

from .config import PredictorConfig
from .types import RoutingStep

TransitionKey = tuple[int, int]


class OnlineExpertPredictor:
    """Small online predictor combining popularity, temporal and cross-layer signals."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self._frequency: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self._temporal: defaultdict[TransitionKey, Counter[int]] = defaultdict(Counter)
        self._cross_layer: defaultdict[TransitionKey, Counter[int]] = defaultdict(
            Counter
        )
        self._last_by_layer: dict[int, tuple[int, ...]] = {}

    def _increment_bounded(
        self,
        table: defaultdict[TransitionKey, Counter[int]],
        key: TransitionKey,
        targets: tuple[int, ...],
    ) -> None:
        counter = table[key]
        limit = self.config.max_targets_per_source
        for target in targets:
            if target in counter:
                counter[target] += 1
                continue
            if len(counter) < limit:
                counter[target] = 1
                continue

            # Space-Saving keeps the table strictly bounded while still letting
            # a sustained new target replace the current least-frequent entry.
            victim = min(counter, key=lambda expert: (counter[expert], expert))
            replacement_count = counter.pop(victim) + 1
            counter[target] = replacement_count

    def observe(self, step: RoutingStep, previous_step: RoutingStep | None) -> None:
        self._frequency[step.layer_index].update(step.experts)

        previous_same_layer = self._last_by_layer.get(step.layer_index)
        if previous_same_layer:
            for source in previous_same_layer:
                self._increment_bounded(
                    self._temporal,
                    (step.layer_index, source),
                    step.experts,
                )

        if (
            previous_step is not None
            and previous_step.token_index == step.token_index
            and previous_step.layer_index + 1 == step.layer_index
        ):
            for source in previous_step.experts:
                self._increment_bounded(
                    self._cross_layer,
                    (step.layer_index, source),
                    step.experts,
                )

        self._last_by_layer[step.layer_index] = step.experts

    def predict(
        self,
        target_layer: int,
        current_step: RoutingStep | None,
        limit: int,
    ) -> tuple[int, ...]:
        if limit <= 0:
            return ()

        scores: Counter[int] = Counter()
        frequency = self._frequency.get(target_layer)
        if frequency and self.config.frequency_weight > 0:
            for expert, count in frequency.items():
                scores[expert] += self.config.frequency_weight * count

        previous_same_layer = self._last_by_layer.get(target_layer)
        if previous_same_layer and self.config.temporal_weight > 0:
            for source in previous_same_layer:
                transitions = self._temporal.get((target_layer, source))
                if transitions:
                    for expert, count in transitions.items():
                        scores[expert] += self.config.temporal_weight * count
                # A cheap persistence prior is useful before transitions are trained.
                scores[source] += self.config.temporal_weight

        if (
            current_step is not None
            and current_step.layer_index + 1 == target_layer
            and self.config.cross_layer_weight > 0
        ):
            for source in current_step.experts:
                transitions = self._cross_layer.get((target_layer, source))
                if transitions:
                    for expert, count in transitions.items():
                        scores[expert] += self.config.cross_layer_weight * count

        ranked = scores.most_common()
        if not ranked:
            return ()
        max_score = float(ranked[0][1])
        if max_score <= 0:
            return ()
        threshold = self.config.min_relative_confidence
        selected: list[int] = []
        for expert, score in ranked:
            if score <= 0:
                continue
            if float(score) / max_score < threshold:
                continue
            selected.append(expert)
            if len(selected) >= limit:
                break
        return tuple(selected)
