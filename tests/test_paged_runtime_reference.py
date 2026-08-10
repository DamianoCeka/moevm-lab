from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class PagedRuntimeReferenceTests(unittest.TestCase):
    def test_reference_is_sanitized_and_comparisons_are_derived(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference_path = (
            root
            / "benchmarks"
            / "reference"
            / "paged-runtime-olmoe-p310-smoke"
            / "result.json"
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))

        serialized = json.dumps(reference)
        for private_value in ('"hostname"', "C:\\\\", "D:\\\\", "Users\\\\"):
            self.assertNotIn(private_value, serialized)

        correctness = reference["correctness"]
        self.assertTrue(correctness["all_match"])
        self.assertEqual(
            correctness["baseline_generated_token_ids"],
            correctness["paged_cold_generated_token_ids"],
        )
        self.assertEqual(
            correctness["baseline_generated_token_ids"],
            correctness["paged_retained_generated_token_ids"],
        )

        measurements = reference["measurements"]
        baseline = measurements["accelerate_cpu_offload_baseline"]
        cold = measurements["paged_lru32_cold_expert_cache"]
        retained = measurements["paged_lru32_retained_expert_cache"]
        comparisons = reference["comparisons_to_baseline"]

        self.assertAlmostEqual(
            baseline["model_load_seconds"] / cold["model_load_seconds"],
            comparisons["paged_model_load_speedup"],
            places=12,
        )
        self.assertAlmostEqual(
            1 - cold["peak_vram_allocated_bytes"] / baseline["peak_vram_bytes"],
            comparisons["peak_vram_reduction_fraction"],
            places=12,
        )
        for label, paged in (("cold", cold), ("retained", retained)):
            expected = comparisons[label]
            end_to_end_speedup = (
                paged["end_to_end_generated_tokens_per_second_including_prefill"]
                / baseline["end_to_end_generated_tokens_per_second_including_prefill"]
            )
            self.assertAlmostEqual(
                baseline["prefill_seconds"] / paged["prefill_seconds"],
                expected["prefill_speedup"],
                places=12,
            )
            self.assertAlmostEqual(
                paged["generation_equivalent_decode_tokens_per_second"]
                / baseline["generation_equivalent_decode_tokens_per_second"],
                expected["decode_throughput_speedup"],
                places=12,
            )
            self.assertAlmostEqual(
                end_to_end_speedup,
                expected["end_to_end_speedup"],
                places=12,
            )
            self.assertAlmostEqual(
                end_to_end_speedup - 1,
                expected["end_to_end_throughput_change_fraction"],
                places=12,
            )

        self.assertLess(comparisons["cold"]["end_to_end_speedup"], 1)
        self.assertGreater(comparisons["retained"]["end_to_end_speedup"], 1)
        self.assertGreater(comparisons["peak_vram_reduction_fraction"], 0)

        expert_bytes = reference["model"]["expert_bytes"]
        policy = reference["runtime_policy"]
        self.assertEqual(
            policy["expert_cache_bytes"],
            reference["model"]["layers"] * policy["slots_per_layer"] * expert_bytes,
        )
        self.assertEqual(
            policy["expected_weight_vram_bytes"],
            policy["expert_cache_bytes"] + policy["non_expert_checkpoint_bytes"],
        )
        for paged in (cold, retained):
            self.assertEqual(
                paged["total_requests"], paged["total_hits"] + paged["total_misses"]
            )
            self.assertAlmostEqual(
                paged["total_hits"] / paged["total_requests"],
                paged["total_hit_rate"],
                places=12,
            )
            self.assertEqual(
                paged["logical_storage_bytes"],
                paged["total_misses"] * expert_bytes,
            )
            self.assertEqual(
                paged["logical_host_to_device_bytes"],
                paged["logical_storage_bytes"],
            )
        retained_over_cold = comparisons["retained_over_cold"]
        self.assertAlmostEqual(
            retained["end_to_end_generated_tokens_per_second_including_prefill"]
            / cold["end_to_end_generated_tokens_per_second_including_prefill"],
            retained_over_cold["end_to_end_speedup"],
            places=12,
        )
        traffic_ratio = (
            retained["logical_storage_bytes"] / cold["logical_storage_bytes"]
        )
        self.assertAlmostEqual(
            traffic_ratio,
            retained_over_cold["logical_storage_traffic_ratio"],
            places=12,
        )
        self.assertAlmostEqual(
            1 - traffic_ratio,
            retained_over_cold["logical_storage_traffic_reduction_fraction"],
            places=12,
        )

        raw_artifacts = reference["raw_artifacts_sha256"]
        for relative_path, expected_digest in raw_artifacts.items():
            self.assertFalse(Path(relative_path).is_absolute())
            self.assertEqual(len(expected_digest), 64)
            raw_path = root / relative_path
            if raw_path.is_file():
                actual_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                self.assertEqual(actual_digest, expected_digest)


if __name__ == "__main__":
    unittest.main()
