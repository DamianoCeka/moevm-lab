from __future__ import annotations

import unittest

from moevm.config import PredictorConfig
from moevm.predictor import OnlineExpertPredictor
from moevm.types import RoutingStep


class PredictorTests(unittest.TestCase):
    def test_learns_cross_layer_transition(self) -> None:
        predictor = OnlineExpertPredictor(
            PredictorConfig(
                enabled=True,
                prefetch_count=2,
                frequency_weight=0.0,
                temporal_weight=0.0,
                cross_layer_weight=1.0,
                max_targets_per_source=8,
                min_relative_confidence=0.0,
                deadline_aware=True,
            )
        )

        for token in range(5):
            first = RoutingStep(token, 0, (1, 2))
            second = RoutingStep(token, 1, (7, 8))
            predictor.observe(first, None)
            predictor.observe(second, first)

        current = RoutingStep(6, 0, (1, 2))
        prediction = predictor.predict(target_layer=1, current_step=current, limit=2)
        self.assertEqual(set(prediction), {7, 8})

    def test_target_table_respects_configured_hard_limit(self) -> None:
        predictor = OnlineExpertPredictor(
            PredictorConfig(
                enabled=True,
                prefetch_count=4,
                frequency_weight=0.0,
                temporal_weight=0.0,
                cross_layer_weight=1.0,
                max_targets_per_source=1,
                min_relative_confidence=0.0,
                deadline_aware=True,
            )
        )

        for token, target in enumerate((2, 3, 4)):
            first = RoutingStep(token, 0, (1,))
            second = RoutingStep(token, 1, (target,))
            predictor.observe(first, None)
            predictor.observe(second, first)

        prediction = predictor.predict(
            target_layer=1,
            current_step=RoutingStep(3, 0, (1,)),
            limit=4,
        )
        self.assertLessEqual(len(prediction), 1)

    def test_bounded_target_table_adapts_to_sustained_new_target(self) -> None:
        predictor = OnlineExpertPredictor(
            PredictorConfig(
                enabled=True,
                prefetch_count=1,
                frequency_weight=0.0,
                temporal_weight=0.0,
                cross_layer_weight=1.0,
                max_targets_per_source=1,
                min_relative_confidence=0.0,
                deadline_aware=True,
            )
        )

        for token, target in [(0, 2), *((index, 3) for index in range(1, 101))]:
            first = RoutingStep(token, 0, (1,))
            second = RoutingStep(token, 1, (target,))
            predictor.observe(first, None)
            predictor.observe(second, first)

        prediction = predictor.predict(
            target_layer=1,
            current_step=RoutingStep(101, 0, (1,)),
            limit=1,
        )

        self.assertEqual(prediction, (3,))
        self.assertEqual(len(predictor._cross_layer[(1, 1)]), 1)


if __name__ == "__main__":
    unittest.main()
