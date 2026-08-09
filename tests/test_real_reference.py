from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from moevm.analysis import analyze_routing_trace
from moevm.config import load_config
from moevm.simulator import compare_experiment
from moevm.trace import read_trace


class RealRoutingReferenceTests(unittest.TestCase):
    def test_all_trace_hashes_analyses_and_replays_are_reproducible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference = root / "benchmarks" / "reference" / "real-routing-olmoe-m1"
        study = json.loads((reference / "study.json").read_text(encoding="utf-8"))
        config = load_config(root / "configs" / "olmoe_1b_7b_0924.toml")

        self.assertEqual(study["aggregate"]["records"], 10)
        self.assertEqual(study["aggregate"]["tokens"], 438)
        self.assertEqual(study["aggregate"]["expert_accesses"], 56_064)

        for record in study["records"]:
            trace_path = reference / record["trace"]["path"]
            digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            self.assertEqual(digest, record["trace"]["sha256"])
            steps = read_trace(trace_path)
            analysis = analyze_routing_trace(
                steps,
                experts_per_layer=config.model.experts_per_layer,
                predictor_config=config.predictor,
            )
            self.assertEqual(analysis.score_coverage, 1.0)
            self.assertEqual(analysis.tokens, record["tokens"])
            self.assertAlmostEqual(
                analysis.mean_temporal_overlap,
                record["router"]["mean_temporal_overlap"],
            )
            result = compare_experiment(config, trace=steps)
            self.assertAlmostEqual(result.speedup, record["replay"]["speedup"])
            self.assertAlmostEqual(
                result.ram_to_vram_traffic_change,
                record["replay"]["ram_to_vram_traffic_change"],
            )


if __name__ == "__main__":
    unittest.main()
