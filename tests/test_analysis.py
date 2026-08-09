from __future__ import annotations

import unittest

from moevm.analysis import analyze_routing_trace
from moevm.config import PredictorConfig
from moevm.types import RoutingStep


def _predictor_config() -> PredictorConfig:
    return PredictorConfig(
        enabled=True,
        prefetch_count=2,
        frequency_weight=0.0,
        temporal_weight=5.0,
        cross_layer_weight=1.0,
        max_targets_per_source=8,
        min_relative_confidence=0.0,
        deadline_aware=True,
    )


class TraceAnalysisTests(unittest.TestCase):
    def test_measures_scored_complete_trace(self) -> None:
        steps = [
            RoutingStep(0, 0, (0, 1), (0.55, 0.25)),
            RoutingStep(0, 1, (1, 2), (0.45, 0.35)),
            RoutingStep(1, 0, (0, 1), (0.60, 0.20)),
            RoutingStep(1, 1, (1, 3), (0.50, 0.20)),
            RoutingStep(2, 0, (0, 2), (0.45, 0.30)),
            RoutingStep(2, 1, (1, 3), (0.48, 0.22)),
        ]

        analysis = analyze_routing_trace(
            steps,
            experts_per_layer=4,
            predictor_config=_predictor_config(),
        )

        self.assertEqual(analysis.tokens, 3)
        self.assertEqual(analysis.layers, 2)
        self.assertEqual(analysis.top_k, 2)
        self.assertEqual(analysis.expert_accesses, 12)
        self.assertEqual(analysis.score_coverage, 1.0)
        self.assertAlmostEqual(
            analysis.mean_topk_probability_mass or 0.0,
            0.7583333333333333,
        )
        self.assertAlmostEqual(analysis.mean_temporal_overlap, 0.75)
        self.assertAlmostEqual(analysis.exact_repeat_rate, 0.5)
        self.assertIsNotNone(analysis.predictor)
        assert analysis.predictor is not None
        self.assertEqual(analysis.predictor.evaluated_steps, 4)
        self.assertGreater(analysis.predictor.coverage, 0.0)
        self.assertGreaterEqual(analysis.predictor.precision, 0.0)
        self.assertLessEqual(analysis.predictor.precision, 1.0)

    def test_rejects_incomplete_token_layer_grid(self) -> None:
        steps = [
            RoutingStep(0, 0, (0, 1)),
            RoutingStep(0, 1, (1, 2)),
            RoutingStep(1, 0, (0, 1)),
        ]

        with self.assertRaisesRegex(ValueError, "every token/layer pair"):
            analyze_routing_trace(steps, experts_per_layer=4)

    def test_rejects_duplicate_steps(self) -> None:
        steps = [
            RoutingStep(0, 0, (0, 1)),
            RoutingStep(0, 0, (0, 2)),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_routing_trace(steps, experts_per_layer=4)


if __name__ == "__main__":
    unittest.main()
