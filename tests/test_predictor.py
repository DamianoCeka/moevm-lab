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


if __name__ == "__main__":
    unittest.main()
