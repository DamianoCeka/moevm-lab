from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from moevm.pipeline_profile import (
    PIPELINE_PRIMITIVES,
    build_measured_profile,
    load_pipeline_profile,
    result_binding,
    validate_profile_binding,
)


def _metrics() -> dict[str, int | float]:
    payload: dict[str, int | float] = {
        "requests": 10,
        "hits": 4,
        "misses": 6,
        "evictions": 2,
        "storage_loads": 6,
        "transfer_loads": 6,
        "storage_bytes": 72,
        "host_to_device_bytes": 72,
        "coalesced_requests": 0,
        "admission_rejections": 0,
        "storage_failures": 0,
        "transfer_failures": 0,
    }
    return payload


def _result(
    pipeline: str,
    *,
    cold_seconds: float,
    retained_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "ok",
        "evidence": {"publishable_benchmark_evidence": True},
        "source": {
            "tree_clean": True,
            "commit": "f" * 40,
            "provenance_mode": "benchmark_evidence",
            "benchmark_script_sha256": "a" * 64,
            "paged_runtime_sha256": "b" * 64,
        },
        "reference_comparison": {
            "available": True,
            "matched": True,
            "mode": "teacher_forced",
            "generated_token_ids": [1, 2],
        },
        "model": {
            "model_id": "model",
            "revision": "c" * 40,
            "dtype": "torch.bfloat16",
            "layers": 2,
            "experts_per_layer": 4,
            "top_k": 2,
            "shards": {"model.safetensors": {"sha256": "d" * 64}},
        },
        "runtime": {
            "pipeline": pipeline,
            "device_uuid": "gpu-1",
            "device_name": "Test GPU",
            "policy": "lru",
            "capacity_scope": "independent per-layer partitions",
            "hotset_sha256": None,
            "budget": {
                "slots_per_layer": 2,
                "layers": 2,
                "expert_bytes": 12,
                "cache_bytes": 48,
                "staging_slots": 2,
                "staging_host_bytes": 24,
                "io_workers": 1,
                "non_expert_checkpoint_bytes": 100,
                "expected_weight_vram_bytes": 148,
                "device_total_vram_bytes": 1000,
            },
            "final_metrics": _metrics(),
        },
        "workload": {
            "id": "workload",
            "prompt_sha256": "e" * 64,
            "input_ids": [1, 2, 3],
            "input_tokens": 3,
            "max_new_tokens": 32,
            "decoding": "teacher-forced reference with greedy predictions",
            "seed": 17,
        },
        "environment": {
            "python": "3.12.8",
            "platform": "test-platform",
            "packages": {"torch": "test"},
        },
        "passes": {
            "cold_expert_cache": {
                "total_wall_seconds": cold_seconds,
                "generated_ids": [1, 2],
                "fed_token_ids": [1, 2],
                "teacher_forced": True,
                "metrics": _metrics(),
            },
            "repeat_retained_expert_cache": {
                "total_wall_seconds": retained_seconds,
                "generated_ids": [1, 2],
                "fed_token_ids": [1, 2],
                "teacher_forced": True,
                "metrics": _metrics(),
            },
        },
    }


class PipelineProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _pairs(self) -> list[tuple[dict[str, object], dict[str, object]]]:
        return [
            (
                _result("sync", cold_seconds=10.0 + index, retained_seconds=6.0),
                _result("async", cold_seconds=8.0 + index, retained_seconds=6.2),
            )
            for index in (0.0, 0.1, 0.2)
        ]

    def test_profile_selects_only_consistent_material_async_gain(self) -> None:
        profile = build_measured_profile(self._pairs(), minimum_gain=0.03)

        self.assertEqual(profile["selection"]["cold_expert_cache"], "async")
        self.assertEqual(profile["selection"]["repeat_retained_expert_cache"], "sync")
        self.assertEqual(profile["calibration"]["pairs"], 3)
        self.assertEqual(profile["binding"], result_binding(self._pairs()[0][0]))

    def test_profile_load_hash_and_binding_are_fail_closed(self) -> None:
        profile = build_measured_profile(self._pairs())
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile, sort_keys=True), encoding="utf-8")

        loaded, digest = load_pipeline_profile(path)

        self.assertEqual(loaded, profile)
        self.assertEqual(len(digest), 64)
        validate_profile_binding(loaded, result_binding(self._pairs()[0][0]))
        wrong = copy.deepcopy(result_binding(self._pairs()[0][0]))
        wrong["hardware"]["device_uuid"] = "different"
        with self.assertRaisesRegex(ValueError, "hardware.device_uuid"):
            validate_profile_binding(loaded, wrong)

    def test_profile_rejects_insufficient_or_noncomparable_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            build_measured_profile(self._pairs()[:2])

        pairs = self._pairs()
        async_metrics = pairs[1][1]["passes"]["cold_expert_cache"]["metrics"]
        async_metrics[PIPELINE_PRIMITIVES[0]] += 1
        with self.assertRaisesRegex(ValueError, "cache/traffic primitives"):
            build_measured_profile(pairs)

    def test_profile_rejects_invalid_shape(self) -> None:
        path = self.root / "bad.json"
        path.write_text('{"schema_version": 1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "kind"):
            load_pipeline_profile(path)

    def test_teacher_forced_gate_allows_prediction_mismatch_but_not_wrong_feed(
        self,
    ) -> None:
        pairs = self._pairs()
        for sync_result, async_result in pairs:
            sync_result["reference_comparison"]["matched"] = False
            async_result["reference_comparison"]["matched"] = False
        profile = build_measured_profile(pairs)
        self.assertEqual(profile["calibration"]["pairs"], 3)

        pairs[0][1]["passes"]["cold_expert_cache"]["fed_token_ids"] = [9, 9]
        with self.assertRaisesRegex(ValueError, "exact reference IDs"):
            build_measured_profile(pairs)

    def test_autoregressive_gate_still_requires_exact_reference_match(self) -> None:
        pairs = self._pairs()
        for sync_result, async_result in pairs:
            for result in (sync_result, async_result):
                result["reference_comparison"]["mode"] = "autoregressive_exact_gate"
                result["reference_comparison"]["matched"] = False
                for measured_pass in result["passes"].values():
                    measured_pass["teacher_forced"] = False
        with self.assertRaisesRegex(ValueError, "autoregressive"):
            build_measured_profile(pairs)


if __name__ == "__main__":
    unittest.main()
