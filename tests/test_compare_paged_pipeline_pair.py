from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "compare_paged_pipeline_pair.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_paged_pipeline_pair", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import pair comparator: {_SCRIPT}")
_COMPARATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPARATOR)


class PagedPipelinePairComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    @staticmethod
    def _metrics(*, requests: int, hits: int, evictions: int) -> dict[str, Any]:
        misses = requests - hits
        result: dict[str, Any] = {
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
            "storage_loads": misses,
            "transfer_loads": misses,
            "storage_bytes": misses * 10,
            "host_to_device_bytes": misses * 10,
            "storage_failures": 0,
            "transfer_failures": 0,
            "admission_rejections": 0,
            "coalesced_requests": 0,
        }
        result["hit_rate"] = hits / requests if requests else 0.0
        result.update(
            {
                "storage_seconds": 0.0,
                "transfer_seconds": 0.0,
                "forward_seconds": 0.0,
                "storage_queue_seconds": 0.0,
                "demand_wait_seconds": 0.0,
                "staging_waits": 0,
            }
        )
        return result

    def _pass(self, wall_seconds: float) -> dict[str, Any]:
        first = self._metrics(requests=3, hits=1, evictions=1)
        decode = self._metrics(requests=2, hits=1, evictions=1)
        total = {
            field: first[field] + decode[field]
            for field in _COMPARATOR._CACHE_PRIMITIVES
        }
        total.update({field: 0 for field in _COMPARATOR._ZERO_COUNTERS})
        total["hit_rate"] = total["hits"] / total["requests"]
        total.update(
            {
                "storage_seconds": 0.1,
                "transfer_seconds": 0.1,
                "forward_seconds": 0.2,
                "storage_queue_seconds": 0.0,
                "demand_wait_seconds": 0.0,
                "staging_waits": 0,
            }
        )
        return {
            "total_wall_seconds": wall_seconds,
            "teacher_forced": True,
            "generated_token_count": 2,
            "generated_ids": [101, 102],
            "fed_token_ids": [201, 202],
            "cuda_memory": {
                "peak_allocated_bytes": 1000,
                "peak_reserved_bytes": 1100,
            },
            "metrics": total,
            "prefill": {
                "input_tokens": 3,
                "wall_seconds": 0.5,
                "metrics": copy.deepcopy(first),
            },
            "first_token": {
                "index": 0,
                "predicted_token_id": 101,
                "fed_token_id": 201,
                "metrics": copy.deepcopy(first),
            },
            "decode": {
                "token_count": 1,
                "wall_seconds": wall_seconds - 0.5,
                "per_token": [
                    {
                        "index": 1,
                        "predicted_token_id": 102,
                        "fed_token_id": 202,
                        "metrics": decode,
                    }
                ],
            },
        }

    def _report(self, pipeline: str) -> dict[str, Any]:
        cold_wall, retained_wall = (10.0, 6.0) if pipeline == "sync" else (8.0, 4.5)
        pass_total = self._metrics(requests=5, hits=2, evictions=2)
        final = {
            field: pass_total[field] * 2 for field in _COMPARATOR._CACHE_PRIMITIVES
        }
        final.update({field: 0 for field in _COMPARATOR._ZERO_COUNTERS})
        final["hit_rate"] = final["hits"] / final["requests"]
        final.update(
            {
                "storage_seconds": 0.2,
                "transfer_seconds": 0.2,
                "forward_seconds": 0.4,
                "storage_queue_seconds": 0.0,
                "demand_wait_seconds": 0.0,
                "staging_waits": 0,
            }
        )
        preload = self._metrics(requests=0, hits=0, evictions=0)
        digest = "a" * 64
        return {
            "schema_version": 1,
            "status": "ok",
            "evidence": {
                "label": "test benchmark evidence",
                "publishable_benchmark_evidence": True,
            },
            "source": {
                "commit": "1" * 40,
                "tree_clean": True,
                "benchmark_script": "scripts/benchmark_paged_olmoe.py",
                "benchmark_script_sha256": "2" * 64,
            },
            "model": {
                "model_id": "example/model",
                "revision": "3" * 40,
                "shards": {
                    "model-00001.safetensors": {
                        "sha256": "4" * 64,
                        "size_bytes": 123,
                    }
                },
            },
            "environment": {
                "python": "3.12.8",
                "platform": "Windows-test",
                "packages": {"torch": "test"},
            },
            "model_load": {"static_preload_metrics": preload},
            "runtime": {
                "device": "cuda:0",
                "device_uuid": "test-uuid",
                "device_name": "test GPU",
                "policy": "lru",
                "pipeline": pipeline,
                "capacity_scope": "independent per-layer partitions",
                "hotset_json": None,
                "hotset_sha256": None,
                "protected_hot_per_layer": {"0": 0},
                "budget": {
                    "pipeline": pipeline,
                    "expert_bytes": 10,
                    "slots_per_layer": 8,
                    "staging_slots": 2,
                    "cache_bytes": 1234,
                },
                "final_metrics": {
                    **final,
                    "peak_staging_in_use": 1,
                    "pending_loads_peak": 1,
                },
            },
            "workload": {
                "id": "python_code",
                "prompt_sha256": "5" * 64,
                "input_ids": [7, 8, 9],
                "input_tokens": 3,
                "max_new_tokens": 2,
                "decoding": "teacher-forced reference with greedy predictions",
                "seed": 17,
            },
            "passes": {
                "cold_expert_cache": self._pass(cold_wall),
                "repeat_retained_expert_cache": self._pass(retained_wall),
            },
            "reference_comparison": {
                "available": True,
                "matched": False,
                "mode": "teacher_forced",
                "sha256": digest,
                "generated_token_ids": [201, 202],
                "source_generated_token_count": 2,
                "temperature": 0.0,
            },
        }

    @staticmethod
    def _cuda_timeline() -> dict[str, Any]:
        spans = [
            {
                "lane": "h2d",
                "name": "h2d:0:L0:E1",
                "sequence": 0,
                "layer": 0,
                "expert": 1,
                "start_ms": 0.0,
                "end_ms": 2.0,
                "duration_ms": 2.0,
            },
            {
                "lane": "expert_compute",
                "name": "expert_compute:1:L0:E2",
                "sequence": 1,
                "layer": 0,
                "expert": 2,
                "start_ms": 1.0,
                "end_ms": 3.0,
                "duration_ms": 2.0,
            },
        ]
        summary = _COMPARATOR.summarize_cuda_timeline(
            transfers=[_COMPARATOR.CudaInterval("h2d:0:L0:E1", 0.0, 2.0)],
            compute=[_COMPARATOR.CudaInterval("expert_compute:1:L0:E2", 1.0, 3.0)],
        )
        summary.pop("intervals")
        return {
            "schema_version": 1,
            "status": "measured",
            "method": "cuda_events_v1",
            "scope": "paged_expert_h2d_vs_expert_compute",
            "unit": "milliseconds",
            "complete": True,
            "reason": None,
            "spans": spans,
            "summary": summary,
        }

    @staticmethod
    def _cuda_overlap(calls: list[dict[str, Any]]) -> dict[str, Any]:
        h2d_active = sum(
            call["summary"]["transfer"]["active_duration_ms"] for call in calls
        )
        compute_active = sum(
            call["summary"]["compute"]["active_duration_ms"] for call in calls
        )
        overlap = sum(call["summary"]["overlap"]["duration_ms"] for call in calls)
        return {
            "schema_version": 1,
            "status": "measured",
            "method": "cuda_events_v1",
            "scope": "paged_expert_h2d_vs_expert_compute",
            "unit": "milliseconds",
            "model_call_count": len(calls),
            "measured_model_call_count": len(calls),
            "h2d_interval_count": sum(
                call["summary"]["transfer"]["interval_count"] for call in calls
            ),
            "expert_compute_interval_count": sum(
                call["summary"]["compute"]["interval_count"] for call in calls
            ),
            "h2d_union_ms": h2d_active,
            "expert_compute_union_ms": compute_active,
            "overlap_ms": overlap,
            "h2d_overlap_fraction": overlap / h2d_active,
            "expert_compute_overlap_fraction": overlap / compute_active,
            "h2d_hidden_by_compute_ms": overlap,
            "h2d_exposed_ms": h2d_active - overlap,
            "active_duration_saved_by_overlap_ms": overlap,
            "reason": None,
            "aggregation": "Summed independent model calls.",
        }

    def _with_cuda_telemetry(self, report: dict[str, Any]) -> dict[str, Any]:
        report["runtime"]["cuda_overlap_telemetry"] = {
            "requested": True,
            "method": "cuda_events_v1",
            "scope": "paged_expert_h2d_vs_expert_compute",
        }
        for pass_payload in report["passes"].values():
            prefill = self._cuda_timeline()
            decode = self._cuda_timeline()
            pass_payload["prefill"]["cuda_event_timeline"] = prefill
            pass_payload["first_token"]["cuda_event_timeline"] = copy.deepcopy(prefill)
            pass_payload["decode"]["per_token"][0]["cuda_event_timeline"] = decode
            pass_payload["cuda_overlap"] = self._cuda_overlap([prefill, decode])
        return report

    def _write(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True), encoding="utf-8"
        )
        return path

    def test_valid_pair_reports_separate_cold_and_retained_savings(self) -> None:
        result = _COMPARATOR.compare_reports(
            self._report("sync"), self._report("async")
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], "ok")
        cold = result["passes"]["cold_expert_cache"]
        self.assertEqual(cold["sync_wall_seconds"], 10.0)
        self.assertEqual(cold["async_wall_seconds"], 8.0)
        self.assertEqual(cold["saving_seconds"], 2.0)
        self.assertAlmostEqual(cold["saving_fraction"], 0.2)
        retained = result["passes"]["repeat_retained_expert_cache"]
        self.assertEqual(retained["saving_seconds"], 1.5)
        self.assertAlmostEqual(retained["saving_fraction"], 0.25)

    def test_mode_specific_resolved_pipeline_budget_is_not_an_identity_mismatch(
        self,
    ) -> None:
        sync = self._report("sync")
        async_ = self._report("async")
        for pass_name in _COMPARATOR.PASS_NAMES:
            sync["runtime"]["budget"].setdefault("resolved_pipeline_by_pass", {})[
                pass_name
            ] = "sync"
            async_["runtime"]["budget"].setdefault("resolved_pipeline_by_pass", {})[
                pass_name
            ] = "async"

        result = _COMPARATOR.compare_reports(sync, async_)

        self.assertEqual(result["status"], "ok")

    def test_cuda_overlap_telemetry_setting_must_match_within_a_pair(self) -> None:
        sync = self._report("sync")
        async_ = self._report("async")
        sync["runtime"]["cuda_overlap_telemetry"] = {
            "requested": True,
            "method": "cuda_events_v1",
            "scope": "paged_expert_h2d_vs_expert_compute",
        }
        async_["runtime"]["cuda_overlap_telemetry"] = {
            "requested": False,
            "method": None,
            "scope": None,
        }

        with self.assertRaisesRegex(ValueError, "cuda_overlap_telemetry"):
            _COMPARATOR.compare_reports(sync, async_)

    def test_requested_cuda_telemetry_requires_complete_derived_pass_data(self) -> None:
        sync = self._with_cuda_telemetry(self._report("sync"))
        async_ = self._with_cuda_telemetry(self._report("async"))

        result = _COMPARATOR.compare_reports(sync, async_)

        self.assertEqual(result["status"], "ok")
        del async_["passes"]["cold_expert_cache"]["cuda_overlap"]
        with self.assertRaisesRegex(ValueError, "cuda_overlap must be an object"):
            _COMPARATOR.compare_reports(sync, async_)

    def test_requested_cuda_telemetry_rejects_incomplete_or_tampered_data(self) -> None:
        mutators: dict[str, Callable[[dict[str, Any]], None]] = {
            "incomplete capture": lambda row: row["passes"]["cold_expert_cache"][
                "prefill"
            ]["cuda_event_timeline"].__setitem__("complete", False),
            "malformed span": lambda row: row["passes"]["cold_expert_cache"]["decode"][
                "per_token"
            ][0]["cuda_event_timeline"]["spans"][0].__setitem__("end_ms", -1.0),
            "aggregate mismatch": lambda row: row["passes"]["cold_expert_cache"][
                "cuda_overlap"
            ].__setitem__("overlap_ms", 99.0),
        }
        sync = self._with_cuda_telemetry(self._report("sync"))
        for name, mutate in mutators.items():
            with self.subTest(name=name):
                async_ = self._with_cuda_telemetry(self._report("async"))
                mutate(async_)
                with self.assertRaises(ValueError):
                    _COMPARATOR.compare_reports(sync, async_)

    def test_tampered_identity_tokens_and_metrics_are_rejected(self) -> None:
        mutators: dict[str, Callable[[dict[str, Any]], None]] = {
            "status": lambda row: row.__setitem__("status", "failed"),
            "demo evidence": lambda row: row["evidence"].__setitem__(
                "publishable_benchmark_evidence", False
            ),
            "demo provenance": lambda row: row["source"].__setitem__(
                "provenance_mode", "demo"
            ),
            "source commit": lambda row: row["source"].__setitem__("commit", "6" * 40),
            "dirty source": lambda row: row["source"].__setitem__("tree_clean", False),
            "script hash": lambda row: row["source"].__setitem__(
                "benchmark_script_sha256", "6" * 64
            ),
            "revision": lambda row: row["model"].__setitem__("revision", "6" * 40),
            "shard hash": lambda row: row["model"]["shards"][
                "model-00001.safetensors"
            ].__setitem__("sha256", "6" * 64),
            "input ids": lambda row: row["workload"]["input_ids"].__setitem__(0, 999),
            "seed": lambda row: row["workload"].__setitem__("seed", 18),
            "policy": lambda row: row["runtime"].__setitem__("policy", "hybrid"),
            "budget": lambda row: row["runtime"]["budget"].__setitem__(
                "slots_per_layer", 9
            ),
            "generated ids": lambda row: row["passes"]["cold_expert_cache"][
                "generated_ids"
            ].__setitem__(0, 999),
            "fed ids": lambda row: row["passes"]["cold_expert_cache"][
                "fed_token_ids"
            ].__setitem__(0, 999),
            "reference": lambda row: row["reference_comparison"].__setitem__(
                "sha256", "6" * 64
            ),
            "final primitive": lambda row: row["runtime"]["final_metrics"].__setitem__(
                "hits", 5
            ),
            "pass primitive": lambda row: row["passes"]["cold_expert_cache"][
                "metrics"
            ].__setitem__("evictions", 3),
            "prefill primitive": lambda row: row["passes"]["cold_expert_cache"][
                "prefill"
            ]["metrics"].__setitem__("evictions", 2),
            "first primitive": lambda row: row["passes"]["cold_expert_cache"][
                "first_token"
            ]["metrics"].__setitem__("evictions", 2),
            "decode primitive": lambda row: row["passes"]["cold_expert_cache"][
                "decode"
            ]["per_token"][0]["metrics"].__setitem__("evictions", 2),
            "coalesced": lambda row: row["passes"]["cold_expert_cache"][
                "metrics"
            ].__setitem__("coalesced_requests", 1),
        }
        sync = self._report("sync")
        for name, mutate in mutators.items():
            with self.subTest(name=name):
                async_ = self._report("async")
                mutate(async_)
                with self.assertRaises(ValueError):
                    _COMPARATOR.compare_reports(sync, async_)

    def test_pipeline_roles_are_not_inferred_or_swapped(self) -> None:
        with self.assertRaisesRegex(ValueError, "pipeline"):
            _COMPARATOR.compare_reports(self._report("async"), self._report("sync"))

    def test_output_is_create_only_and_preserves_existing_file(self) -> None:
        sync_path = self._write("sync.json", self._report("sync"))
        async_path = self._write("async.json", self._report("async"))
        output = self.root / "pair.json"

        self.assertEqual(
            _COMPARATOR.main(
                [str(sync_path), str(async_path), "--output", str(output)]
            ),
            0,
        )
        original = output.read_bytes()
        payload = json.loads(original)
        self.assertEqual(
            payload["inputs"]["sync"]["sha256"],
            hashlib.sha256(sync_path.read_bytes()).hexdigest(),
        )
        with self.assertRaises(FileExistsError):
            _COMPARATOR.main([str(sync_path), str(async_path), "--output", str(output)])
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
