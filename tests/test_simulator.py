from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from moevm.config import (
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    PredictorConfig,
    TraceConfig,
    load_config,
)
from moevm.simulator import compare_experiment, run_experiment
from moevm.trace import SyntheticRoutingTrace
from moevm.types import RoutingStep


class SimulatorTests(unittest.TestCase):
    def test_comparison_uses_identical_trace_and_reports_prefetch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml").with_tokens(64)
        trace = list(SyntheticRoutingTrace(config).generate())
        result = compare_experiment(config, trace=trace)

        self.assertEqual(result.baseline.tokens, result.prefetch.tokens)
        self.assertEqual(
            result.baseline.expert_accesses, result.prefetch.expert_accesses
        )
        self.assertGreater(result.prefetch.prefetch_candidates, 0)
        self.assertGreater(result.prefetch.prefetch_loaded, 0)
        self.assertGreater(result.prefetch.prefetch_useful, 0)
        self.assertLessEqual(
            result.prefetch.prefetch_useful,
            result.prefetch.demand_l1_hits,
        )
        self.assertGreater(
            result.prefetch.demand_l1_hit_rate,
            result.baseline.demand_l1_hit_rate,
        )
        self.assertLess(result.prefetch.total_stall_ms, result.baseline.total_stall_ms)
        self.assertGreater(result.speedup, 1.0)

    def test_prefetch_metrics_count_only_real_demand_hits(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                name="prefetch-accounting",
                layers=1,
                experts_per_layer=4,
                top_k=2,
                expert_size_mib=1.0,
                compute_ms_per_layer=1.0,
            ),
            hardware=HardwareConfig(
                vram_cache_mib=2.0,
                ram_cache_mib=4.0,
                ram_to_vram_gbps=10.0,
                nvme_to_ram_gbps=10.0,
                ram_latency_us=0.0,
                nvme_latency_us=0.0,
                overlap_efficiency=1.0,
                prefetch_vram_fraction=0.75,
            ),
            trace=TraceConfig(
                tokens=2,
                domains=1,
                domain_switch_probability=0.0,
                hotset_multiplier=1.0,
                temporal_reuse_probability=1.0,
                random_expert_probability=0.0,
                seed=1,
            ),
            predictor=PredictorConfig(
                enabled=True,
                prefetch_count=2,
                frequency_weight=1.0,
                temporal_weight=0.0,
                cross_layer_weight=0.0,
                max_targets_per_source=4,
                min_relative_confidence=0.0,
                deadline_aware=False,
            ),
        )
        trace = [
            RoutingStep(0, 0, (0, 1)),
            RoutingStep(1, 0, (0, 1)),
        ]

        metrics = run_experiment(config, mode="prefetch", trace=trace)

        self.assertEqual(metrics.prefetch_loaded, 2)
        self.assertEqual(metrics.prefetch_useful, 1)
        self.assertEqual(metrics.prefetch_wasted, 1)
        self.assertEqual(metrics.demand_l1_hits, 1)

    def test_zero_prefetch_count_is_identical_to_baseline(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                name="prefetch-disabled-by-count",
                layers=1,
                experts_per_layer=2,
                top_k=1,
                expert_size_mib=1.0,
                compute_ms_per_layer=1.0,
            ),
            hardware=HardwareConfig(
                vram_cache_mib=2.0,
                ram_cache_mib=2.0,
                ram_to_vram_gbps=10.0,
                nvme_to_ram_gbps=10.0,
                ram_latency_us=0.0,
                nvme_latency_us=0.0,
                overlap_efficiency=1.0,
                prefetch_vram_fraction=0.5,
            ),
            trace=TraceConfig(
                tokens=10,
                domains=1,
                domain_switch_probability=0.0,
                hotset_multiplier=1.0,
                temporal_reuse_probability=0.0,
                random_expert_probability=0.0,
                seed=1,
            ),
            predictor=PredictorConfig(
                enabled=True,
                prefetch_count=0,
                frequency_weight=1.0,
                temporal_weight=1.0,
                cross_layer_weight=1.0,
                max_targets_per_source=2,
                min_relative_confidence=0.0,
                deadline_aware=True,
            ),
        )
        trace = [RoutingStep(token, 0, (token % 2,)) for token in range(10)]

        result = compare_experiment(config, trace=trace)

        self.assertEqual(result.prefetch.prefetch_predictions, 0)
        self.assertEqual(result.prefetch.prefetch_candidates, 0)
        self.assertEqual(result.prefetch.demand_l1_hits, result.baseline.demand_l1_hits)
        self.assertEqual(result.prefetch.elapsed_ms, result.baseline.elapsed_ms)
        self.assertEqual(result.prefetch.final_cache, result.baseline.final_cache)

    def test_reloaded_prefetch_remains_active_when_resident_after_batch(self) -> None:
        config = ExperimentConfig(
            model=ModelConfig(
                name="prefetch-reload-accounting",
                layers=1,
                experts_per_layer=4,
                top_k=1,
                expert_size_mib=1.0,
                compute_ms_per_layer=1.0,
            ),
            hardware=HardwareConfig(
                vram_cache_mib=2.0,
                ram_cache_mib=4.0,
                ram_to_vram_gbps=10.0,
                nvme_to_ram_gbps=10.0,
                ram_latency_us=0.0,
                nvme_latency_us=0.0,
                overlap_efficiency=1.0,
                prefetch_vram_fraction=0.5,
            ),
            trace=TraceConfig(
                tokens=3,
                domains=1,
                domain_switch_probability=0.0,
                hotset_multiplier=1.0,
                temporal_reuse_probability=0.0,
                random_expert_probability=0.0,
                seed=1,
            ),
            predictor=PredictorConfig(
                enabled=True,
                prefetch_count=2,
                frequency_weight=1.0,
                temporal_weight=0.0,
                cross_layer_weight=0.0,
                max_targets_per_source=4,
                min_relative_confidence=0.0,
                deadline_aware=False,
            ),
        )

        class ScriptedPredictor:
            def __init__(self, _config: PredictorConfig) -> None:
                self._predictions = iter(((1,), (2, 1)))

            def observe(
                self,
                _step: RoutingStep,
                _previous_step: RoutingStep | None,
            ) -> None:
                return None

            def predict(
                self,
                target_layer: int,
                current_step: RoutingStep | None,
                limit: int,
            ) -> tuple[int, ...]:
                return next(self._predictions)

        trace = [
            RoutingStep(0, 0, (0,)),
            RoutingStep(1, 0, (0,)),
            RoutingStep(2, 0, (1,)),
        ]
        with patch("moevm.simulator.OnlineExpertPredictor", ScriptedPredictor):
            metrics = run_experiment(config, mode="prefetch", trace=trace)

        self.assertEqual(metrics.prefetch_loaded, 3)
        self.assertEqual(metrics.prefetch_useful, 1)
        self.assertEqual(metrics.prefetch_wasted, 2)
        self.assertEqual(metrics.prefetch_useful + metrics.prefetch_wasted, 3)

    def test_accepts_complete_trace_with_nonzero_token_offset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml").with_tokens(1)
        trace = [
            RoutingStep(10, layer, (0, 1, 2, 3)) for layer in range(config.model.layers)
        ]

        metrics = run_experiment(config, mode="baseline", trace=trace)

        self.assertEqual(metrics.tokens, 1)
        self.assertEqual(metrics.steps, config.model.layers)

    def test_rejects_incomplete_token_layer_sequence(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml")

        with self.assertRaisesRegex(ValueError, "complete, ordered"):
            run_experiment(
                config,
                trace=[RoutingStep(0, 0, (0, 1, 2, 3))],
            )

    def test_rejects_out_of_order_layers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml")
        steps = [
            RoutingStep(0, layer, (0, 1, 2, 3)) for layer in range(config.model.layers)
        ]
        steps[0], steps[1] = steps[1], steps[0]

        with self.assertRaisesRegex(ValueError, "expected token 0, layer 0"):
            run_experiment(config, trace=steps)

    def test_rejects_unknown_run_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml").with_tokens(1)

        with self.assertRaisesRegex(ValueError, "unknown run mode"):
            run_experiment(config, mode="typo")  # type: ignore[arg-type]

    def test_compare_validates_before_generating_trace(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml")
        invalid = replace(
            config,
            model=replace(config.model, experts_per_layer=1, top_k=2),
        )

        with self.assertRaisesRegex(ValueError, "model.top_k"):
            compare_experiment(invalid)


if __name__ == "__main__":
    unittest.main()
