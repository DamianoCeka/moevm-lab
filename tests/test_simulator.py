from __future__ import annotations

import unittest
from pathlib import Path

from moevm.config import load_config
from moevm.simulator import compare_experiment
from moevm.trace import SyntheticRoutingTrace


class SimulatorTests(unittest.TestCase):
    def test_comparison_uses_identical_trace_and_reports_prefetch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml").with_tokens(64)
        trace = list(SyntheticRoutingTrace(config).generate())
        result = compare_experiment(config, trace=trace)

        self.assertEqual(result.baseline.tokens, result.prefetch.tokens)
        self.assertEqual(result.baseline.expert_accesses, result.prefetch.expert_accesses)
        self.assertGreater(result.prefetch.prefetch_candidates, 0)
        self.assertGreater(result.prefetch.prefetch_loaded, 0)
        self.assertGreater(result.prefetch.prefetch_useful, 0)
        self.assertGreater(
            result.prefetch.demand_l1_hit_rate,
            result.baseline.demand_l1_hit_rate,
        )
        self.assertLess(result.prefetch.total_stall_ms, result.baseline.total_stall_ms)
        self.assertGreater(result.speedup, 1.0)


if __name__ == "__main__":
    unittest.main()
