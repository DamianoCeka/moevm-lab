from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_paged_capacity_sweep.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "summarize_paged_capacity_sweep", _SCRIPT
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import capacity-sweep summarizer: {_SCRIPT}")
_SUMMARY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SUMMARY)


class CapacitySweepSummaryTests(unittest.TestCase):
    CAPACITIES = (16, 24, 32, 40)
    REPETITIONS = 3
    MAX_NEW_TOKENS = 4
    SEED = 17
    SOURCE_COMMIT = "a" * 40

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.workload_file = self.root / "workloads.json"
        self.baseline_dir = self.root / "baseline"
        self.paged_dir = self.root / "paged"
        self.baseline_dir.mkdir()
        self.workloads = [
            {"id": "systems_en", "prompt": "Explain virtual memory briefly."},
            {"id": "python_code", "prompt": "Write a small Python iterator."},
        ]
        self.reference_ids = {
            "systems_en": [1, 2, 3, 4],
            "python_code": [5, 6, 7, 8],
        }
        self.predicted_ids = {
            "systems_en": [1, 99, 3, 98],
            "python_code": [5, 6, 7, 8],
        }
        self._write_json(
            self.workload_file,
            {"schema_version": 1, "workloads": self.workloads},
        )
        self._write_fixture()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AssertionError(f"fixture is not an object: {path}")
        return payload

    def _baseline_path(self, workload_id: str) -> Path:
        return self.baseline_dir / f"{workload_id}.metadata.json"

    def _paged_path(self, capacity: int, workload_id: str, repetition: int = 1) -> Path:
        return (
            self.paged_dir
            / f"repetition-{repetition}"
            / f"slots-{capacity}"
            / f"{workload_id}.json"
        )

    def _baseline_payload(
        self,
        *,
        workload: dict[str, str],
        wall_seconds: float,
        peak_vram_bytes: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": {
                "id": _SUMMARY.PINNED_MODEL_ID,
                "requested_revision": _SUMMARY.PINNED_REVISION,
                "resolved_revision": _SUMMARY.PINNED_REVISION,
                "checkpoint_shards_sha256": _SUMMARY.PINNED_SHARD_SHA256,
            },
            "workload": {
                "id": workload["id"],
                "prompt_sha256": hashlib.sha256(
                    workload["prompt"].encode("utf-8")
                ).hexdigest(),
                "workload_file_sha256": hashlib.sha256(
                    self.workload_file.read_bytes()
                ).hexdigest(),
            },
            "generation": {
                "temperature": 0.0,
                "seed": self.SEED,
                "generated_tokens": self.MAX_NEW_TOKENS,
                "generated_token_ids": self.reference_ids[workload["id"]],
            },
            "timing_observation": {
                "generation_wall_seconds": wall_seconds,
                "prefill_seconds": 2.0,
                "generation_decode_seconds": wall_seconds - 2.0,
            },
            "environment": {"peak_vram_bytes": peak_vram_bytes},
        }

    def _pass_payload(
        self,
        *,
        workload_id: str,
        wall_seconds: float,
        requests: int,
        hits: int,
        peak_vram_bytes: int,
        peak_rss_bytes: int,
    ) -> dict[str, Any]:
        storage_bytes = (requests - hits) * _SUMMARY.PINNED_EXPERT_BYTES
        return {
            "total_wall_seconds": wall_seconds,
            "teacher_forced": True,
            "generated_token_count": self.MAX_NEW_TOKENS,
            "generated_ids": self.predicted_ids[workload_id],
            "fed_token_ids": self.reference_ids[workload_id],
            "metrics": {
                "requests": requests,
                "hits": hits,
                "misses": requests - hits,
                "evictions": max(0, requests - hits - 1),
                "storage_bytes": storage_bytes,
                "host_to_device_bytes": storage_bytes,
                "storage_seconds": 0.2,
                "transfer_seconds": 0.1,
                "forward_seconds": 0.5,
            },
            "prefill": {"wall_seconds": 1.0},
            "decode": {
                "token_count": self.MAX_NEW_TOKENS - 1,
                "wall_seconds": wall_seconds - 1.0,
            },
            "cuda_memory": {
                "peak_allocated_bytes": peak_vram_bytes,
                "peak_reserved_bytes": peak_vram_bytes + 25,
            },
            "process_memory_after": {"peak_rss_bytes": peak_rss_bytes},
        }

    def _paged_payload(
        self,
        *,
        workload: dict[str, str],
        capacity: int,
        repetition: int,
        baseline_sha256: str,
        cold: dict[str, Any],
        retained: dict[str, Any],
    ) -> dict[str, Any]:
        cache_bytes = _SUMMARY.PINNED_LAYERS * capacity * _SUMMARY.PINNED_EXPERT_BYTES
        non_expert_bytes = 1_000_000
        return {
            "schema_version": 1,
            "status": "ok",
            "created_at": (f"2026-08-13T{9 + repetition:02d}:{capacity:02d}:00+00:00"),
            "model": {
                "model_id": _SUMMARY.PINNED_MODEL_ID,
                "revision": _SUMMARY.PINNED_REVISION,
                "shards": {
                    name: {"sha256": digest}
                    for name, digest in _SUMMARY.PINNED_SHARD_SHA256.items()
                },
            },
            "runtime": {
                "policy": "lru",
                "device_name": "Synthetic GPU",
                "budget": {
                    "slots_per_layer": capacity,
                    "staging_slots": 1,
                    "layers": _SUMMARY.PINNED_LAYERS,
                    "expert_bytes": _SUMMARY.PINNED_EXPERT_BYTES,
                    "cache_bytes": cache_bytes,
                    "non_expert_checkpoint_bytes": non_expert_bytes,
                    "expected_weight_vram_bytes": cache_bytes + non_expert_bytes,
                },
            },
            "workload": {
                "id": workload["id"],
                "prompt_sha256": hashlib.sha256(
                    workload["prompt"].encode("utf-8")
                ).hexdigest(),
                "input_tokens": 12,
                "max_new_tokens": self.MAX_NEW_TOKENS,
                "seed": self.SEED,
                "decoding": "teacher-forced reference with greedy predictions",
            },
            "passes": {
                "cold_expert_cache": cold,
                "repeat_retained_expert_cache": retained,
            },
            "reference_comparison": {
                "available": True,
                "mode": "teacher_forced",
                "sha256": baseline_sha256,
                "generated_token_ids": self.reference_ids[workload["id"]],
            },
            "environment": {
                "python": "3.12.0",
                "platform": "synthetic-test",
                "packages": {"torch": "test"},
            },
        }

    def _write_fixture(self) -> None:
        baseline_walls = (10.0, 20.0)
        baseline_hashes: dict[str, str] = {}
        for index, workload in enumerate(self.workloads):
            path = self._baseline_path(workload["id"])
            self._write_json(
                path,
                self._baseline_payload(
                    workload=workload,
                    wall_seconds=baseline_walls[index],
                    peak_vram_bytes=1_000 + index * 1_000,
                ),
            )
            baseline_hashes[workload["id"]] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

        cold_walls = {
            16: (8.0, 8.0),
            24: (7.0, 7.6),
            32: (7.0, 7.0),
            40: (7.5, 7.5),
        }
        retained_walls = {
            16: (5.5, 5.5),
            24: (5.0, 5.0),
            32: (5.25, 5.25),
            40: (4.0, 4.0),
        }
        cold_peak_vram = {16: 100, 24: 140, 32: 190, 40: 240}
        retained_peak_vram = {16: 100, 24: 140, 32: 190, 40: 240}
        for repetition in range(1, self.REPETITIONS + 1):
            wall_offset = (repetition - 2) * 0.5
            for capacity in self.CAPACITIES:
                for index, workload in enumerate(self.workloads):
                    if capacity == 24:
                        cold_requests, cold_hits = ((10, 9), (90, 45))[index]
                    else:
                        cold_requests, cold_hits = (20, 10 + index)
                    cold = self._pass_payload(
                        workload_id=workload["id"],
                        wall_seconds=cold_walls[capacity][index] + wall_offset,
                        requests=cold_requests,
                        hits=cold_hits,
                        peak_vram_bytes=cold_peak_vram[capacity] + index * 10,
                        peak_rss_bytes=1_000 + index * 1_000,
                    )
                    retained = self._pass_payload(
                        workload_id=workload["id"],
                        wall_seconds=retained_walls[capacity][index] + wall_offset,
                        requests=20,
                        hits=18 + index,
                        peak_vram_bytes=retained_peak_vram[capacity] + index * 10,
                        peak_rss_bytes=900 + index * 1_000,
                    )
                    self._write_json(
                        self._paged_path(capacity, workload["id"], repetition),
                        self._paged_payload(
                            workload=workload,
                            capacity=capacity,
                            repetition=repetition,
                            baseline_sha256=baseline_hashes[workload["id"]],
                            cold=cold,
                            retained=retained,
                        ),
                    )

    def _summarize(self) -> dict[str, Any]:
        return _SUMMARY.summarize_capacity_sweep(
            workload_file=self.workload_file,
            baseline_dir=self.baseline_dir,
            paged_dir=self.paged_dir,
            capacities=self.CAPACITIES,
            repetitions=self.REPETITIONS,
            max_new_tokens=self.MAX_NEW_TOKENS,
            seed=self.SEED,
            source_commit=self.SOURCE_COMMIT,
        )

    def _rewrite_baseline_wall_and_rebind(
        self, workload_id: str, wall_seconds: float
    ) -> None:
        baseline_path = self._baseline_path(workload_id)
        baseline = self._read_json(baseline_path)
        baseline["timing_observation"]["generation_wall_seconds"] = wall_seconds
        self._write_json(baseline_path, baseline)
        digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        for repetition in range(1, self.REPETITIONS + 1):
            for capacity in self.CAPACITIES:
                paged_path = self._paged_path(capacity, workload_id, repetition)
                paged = self._read_json(paged_path)
                paged["reference_comparison"]["sha256"] = digest
                self._write_json(paged_path, paged)

    def test_aggregates_primitives_and_uses_weighted_hit_rate(self) -> None:
        report = self._summarize()
        capacity = next(
            row for row in report["capacities"] if row["slots_per_layer"] == 24
        )
        middle_cold = capacity["repetitions"][1]["aggregate"]["cold"]
        cold = capacity["aggregate_across_repetitions"]["cold"]

        self.assertAlmostEqual(report["baseline"]["wall_seconds"], 30.0)
        self.assertAlmostEqual(middle_cold["wall_seconds"], 14.6)
        self.assertAlmostEqual(middle_cold["speedup_over_baseline"], 30.0 / 14.6)
        self.assertAlmostEqual(cold["wall_seconds_median"], 14.6)
        self.assertAlmostEqual(cold["wall_seconds_min"], 13.6)
        self.assertAlmostEqual(cold["wall_seconds_max"], 15.6)
        self.assertAlmostEqual(cold["speedup_over_baseline_median"], 30.0 / 14.6)
        self.assertAlmostEqual(cold["speedup_over_baseline_min"], 30.0 / 15.6)
        self.assertAlmostEqual(cold["speedup_over_baseline_max"], 30.0 / 13.6)
        self.assertEqual(cold["repetitions"], 3)
        self.assertEqual(cold["requests"], 300)
        self.assertEqual(cold["hits"], 162)
        self.assertAlmostEqual(cold["hit_rate"], 0.54)
        self.assertNotAlmostEqual(cold["hit_rate"], (0.9 + 0.5) / 2)
        expected_transfer = 3 * 46 * _SUMMARY.PINNED_EXPERT_BYTES
        self.assertEqual(cold["logical_storage_bytes"], expected_transfer)
        self.assertEqual(cold["host_to_device_bytes"], expected_transfer)
        self.assertEqual(cold["peak_vram_bytes"], 150)
        self.assertEqual(cold["peak_rss_bytes"], 2_000)
        self.assertEqual(middle_cold["decode_tokens"], 6)
        self.assertAlmostEqual(middle_cold["decode_seconds"], 12.6)
        self.assertAlmostEqual(cold["decode_tokens_per_second_median"], 6 / 12.6)
        self.assertEqual(cold["prediction_matches"], 18)
        self.assertEqual(cold["prediction_total"], 24)
        self.assertAlmostEqual(cold["prediction_match_rate"], 0.75)
        self.assertEqual(report["protocol"]["repetitions"], 3)
        self.assertEqual(len(capacity["repetitions"]), 3)
        self.assertIn(
            "3 repetitions provide only a limited estimate of run-to-run variance.",
            report["limitations"],
        )
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_reports_cold_and_retained_pareto_and_balanced_selection(self) -> None:
        report = self._summarize()

        self.assertEqual(report["pareto"]["cold_slots_per_layer"], [16, 24, 32])
        self.assertEqual(report["pareto"]["retained_slots_per_layer"], [16, 24, 40])
        self.assertEqual(report["balanced_selection"]["slots_per_layer"], 24)
        self.assertAlmostEqual(
            report["balanced_selection"]["cold_wall_seconds_median"], 14.6
        )
        self.assertEqual(
            report["balanced_selection"]["cold_wall_seconds_range"], [13.6, 15.6]
        )
        self.assertAlmostEqual(
            report["balanced_selection"]["cold_speedup_over_baseline_median"],
            30.0 / 14.6,
        )

    def test_reports_no_balanced_selection_when_none_beats_baseline(self) -> None:
        for workload in self.workloads:
            self._rewrite_baseline_wall_and_rebind(workload["id"], 4.0)

        report = self._summarize()

        self.assertEqual(
            report["balanced_selection"],
            {
                "eligible": False,
                "reason": "no tested capacity met the predeclared selection rule",
            },
        )

    def test_rejects_tampered_teacher_forced_sequence(self) -> None:
        path = self._paged_path(16, "systems_en")
        payload = self._read_json(path)
        payload["passes"]["cold_expert_cache"]["fed_token_ids"][0] = 999
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "exact reference sequence"):
            self._summarize()

    def test_rejects_cold_retained_prediction_divergence(self) -> None:
        path = self._paged_path(16, "systems_en")
        payload = self._read_json(path)
        payload["passes"]["repeat_retained_expert_cache"]["generated_ids"][0] = 999
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "cold and retained predictions differ"):
            self._summarize()

    def test_rejects_tampered_shard_digest(self) -> None:
        path = self._paged_path(16, "systems_en")
        payload = self._read_json(path)
        shard = next(iter(payload["model"]["shards"].values()))
        shard["sha256"] = "0" * 64
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "shard verification mismatch"):
            self._summarize()

    def test_rejects_inconsistent_request_counters(self) -> None:
        path = self._paged_path(16, "systems_en")
        payload = self._read_json(path)
        payload["passes"]["cold_expert_cache"]["metrics"]["hits"] += 1
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "request counters are inconsistent"):
            self._summarize()

    def test_rejects_inconsistent_transfer_counters(self) -> None:
        path = self._paged_path(16, "systems_en")
        payload = self._read_json(path)
        payload["passes"]["cold_expert_cache"]["metrics"]["host_to_device_bytes"] += 1
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "transfer counters are inconsistent"):
            self._summarize()

    def test_rejects_inconsistent_cache_budget(self) -> None:
        path = self._paged_path(16, "systems_en")
        payload = self._read_json(path)
        payload["runtime"]["budget"]["cache_bytes"] += 1
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "runtime budget mismatch"):
            self._summarize()

    def test_rejects_stale_baseline_binding_after_baseline_tampering(self) -> None:
        path = self._baseline_path("systems_en")
        payload = self._read_json(path)
        payload["timing_observation"]["generation_wall_seconds"] = 11.0
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "reference binding mismatch"):
            self._summarize()

    def test_rejects_non_numeric_greedy_temperature(self) -> None:
        path = self._baseline_path("systems_en")
        payload = self._read_json(path)
        payload["generation"]["temperature"] = False
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "temperature"):
            self._summarize()

    def test_rejects_zero_baseline_wall_time(self) -> None:
        path = self._baseline_path("systems_en")
        payload = self._read_json(path)
        payload["timing_observation"]["generation_wall_seconds"] = 0.0
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "generation_wall_seconds"):
            self._summarize()

    def test_rejects_one_token_protocol_before_reading_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 2 and 64"):
            _SUMMARY.summarize_capacity_sweep(
                workload_file=self.workload_file,
                baseline_dir=self.root / "missing-baseline",
                paged_dir=self.root / "missing-paged",
                capacities=self.CAPACITIES,
                repetitions=self.REPETITIONS,
                max_new_tokens=1,
                seed=self.SEED,
                source_commit=self.SOURCE_COMMIT,
            )

    def test_rejects_mixed_hardware_environments(self) -> None:
        path = self._paged_path(40, "python_code", repetition=3)
        payload = self._read_json(path)
        payload["runtime"]["device_name"] = "Different Synthetic GPU"
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "different hardware/software"):
            self._summarize()

    def test_rejects_mixed_software_environments(self) -> None:
        path = self._paged_path(40, "python_code", repetition=3)
        payload = self._read_json(path)
        payload["environment"]["packages"]["torch"] = "different-version"
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "different hardware/software"):
            self._summarize()

    def test_rejects_malformed_runtime_budget_with_validation_error(self) -> None:
        path = self._paged_path(16, "systems_en")
        payload = self._read_json(path)
        payload["runtime"]["budget"] = copy.deepcopy([{"slots_per_layer": 16}])
        self._write_json(path, payload)

        with self.assertRaisesRegex(ValueError, "runtime budget missing"):
            self._summarize()


if __name__ == "__main__":
    unittest.main()
