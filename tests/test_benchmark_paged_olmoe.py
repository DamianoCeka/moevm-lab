from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_paged_olmoe.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_paged_olmoe", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import benchmark harness: {_SCRIPT}")
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)


class PagedOlmoeHarnessTests(unittest.TestCase):
    def test_process_memory_reports_current_rss_when_supported(self) -> None:
        memory = _HARNESS._process_memory()

        self.assertIn("rss_bytes", memory)
        self.assertIn("peak_rss_bytes", memory)
        if sys.platform in ("win32", "linux"):
            self.assertIsInstance(memory["rss_bytes"], int)
            self.assertGreater(memory["rss_bytes"], 0)
            self.assertIsInstance(memory["peak_rss_bytes"], int)
            self.assertGreaterEqual(memory["peak_rss_bytes"], memory["rss_bytes"])

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def _args(self, *extra: str) -> argparse.Namespace:
        return _HARNESS.build_parser().parse_args(
            [
                "--snapshot",
                str(self.root / _HARNESS.PINNED_REVISION),
                "--output",
                str(self.root / "result.json"),
                *extra,
            ]
        )

    def _reference_payload(self, prompt: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model": {
                "id": _HARNESS.PINNED_MODEL_ID,
                "requested_revision": _HARNESS.PINNED_REVISION,
                "resolved_revision": _HARNESS.PINNED_REVISION,
            },
            "workload": {
                "id": "python_code",
                "prompt_sha256": _HARNESS._prompt_sha256(prompt),
            },
            "generation": {
                "temperature": 0.0,
                "generated_token_ids": [187, 187, 510],
            },
        }

    def test_parser_has_guarded_small_defaults(self) -> None:
        args = self._args()

        _HARNESS._validate_args(args)

        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.policy, "lru")
        self.assertEqual(args.pipeline, "sync")
        self.assertEqual(args.slots_per_layer, 32)
        self.assertEqual(args.staging_slots, 1)
        self.assertEqual(args.max_new_tokens, 2)
        self.assertIsNone(args.prompt)
        self.assertFalse(args.teacher_force_reference)
        self.assertFalse(args.demo_mode)
        self.assertFalse(args.cuda_overlap_telemetry)

        telemetry_args = self._args("--cuda-overlap-telemetry")
        _HARNESS._validate_args(telemetry_args)
        self.assertTrue(telemetry_args.cuda_overlap_telemetry)

        demo_args = self._args("--demo-mode")
        _HARNESS._validate_args(demo_args)
        self.assertTrue(demo_args.demo_mode)

        long_args = self._args("--max-new-tokens", "64")
        _HARNESS._validate_args(long_args)
        self.assertEqual(long_args.max_new_tokens, 64)
        parse_max_new_tokens = _HARNESS._bounded_integer("max-new-tokens", 1, 64)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_max_new_tokens("65")

    def test_policy_and_device_validation_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --hotset-json"):
            _HARNESS._validate_args(self._args("--policy", "hybrid"))
        with self.assertRaisesRegex(ValueError, "only valid"):
            _HARNESS._validate_args(
                self._args("--hotset-json", str(self.root / "hot.json"))
            )
        with self.assertRaisesRegex(ValueError, "device must"):
            _HARNESS._validate_args(self._args("--device", "cpu"))
        with self.assertRaisesRegex(ValueError, "requires --reference-metadata"):
            _HARNESS._validate_args(self._args("--teacher-force-reference"))
        with self.assertRaisesRegex(ValueError, "at least two staging slots"):
            _HARNESS._validate_args(self._args("--pipeline", "async"))
        with self.assertRaisesRegex(ValueError, "at least two staging slots"):
            _HARNESS._validate_args(self._args("--pipeline", "adaptive"))
        with self.assertRaisesRegex(ValueError, "requires --pipeline-profile"):
            _HARNESS._validate_args(
                self._args("--pipeline", "auto", "--staging-slots", "2")
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            _HARNESS._validate_args(
                self._args("--pipeline-profile", str(self.root / "profile.json"))
            )
        with self.assertRaisesRegex(ValueError, "not available"):
            _HARNESS._validate_args(
                self._args(
                    "--pipeline",
                    "auto",
                    "--pipeline-profile",
                    str(self.root / "profile.json"),
                    "--staging-slots",
                    "2",
                    "--demo-mode",
                )
            )
        async_args = self._args(
            "--pipeline",
            "async",
            "--staging-slots",
            "2",
        )
        _HARNESS._validate_args(async_args)
        self.assertEqual(async_args.pipeline, "async")
        adaptive_args = self._args(
            "--pipeline",
            "adaptive",
            "--staging-slots",
            "2",
        )
        _HARNESS._validate_args(adaptive_args)
        self.assertEqual(adaptive_args.pipeline, "adaptive")
        auto_args = self._args(
            "--pipeline",
            "auto",
            "--pipeline-profile",
            str(self.root / "profile.json"),
            "--staging-slots",
            "2",
        )
        _HARNESS._validate_args(auto_args)
        self.assertEqual(auto_args.pipeline, "auto")

    def test_workload_file_selects_exact_prompt(self) -> None:
        workload_path = self.root / "workloads.json"
        workload_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workloads": [
                        {"id": "selected", "prompt": "Selected prompt."},
                        {"id": "other", "prompt": "Other prompt."},
                    ],
                }
            ),
            encoding="utf-8",
        )
        args = self._args(
            "--workload-file",
            str(workload_path),
            "--workload-id",
            "selected",
        )

        self.assertEqual(_HARNESS._resolve_prompt(args), "Selected prompt.")

    def test_streaming_hash_verification_detects_mismatch(self) -> None:
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        shard = snapshot / "shard.safetensors"
        shard.write_bytes(b"paged-runtime-test" * 100)
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()

        verified = _HARNESS._verify_pinned_shards(
            snapshot,
            expected_hashes={shard.name: digest},
        )
        self.assertEqual(verified[shard.name]["sha256"], digest)
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            _HARNESS._verify_pinned_shards(
                snapshot,
                expected_hashes={shard.name: "0" * 64},
            )

    def test_store_is_closed_before_checkpoint_verification(self) -> None:
        events: list[str] = []

        class FakeStore:
            @staticmethod
            def close() -> None:
                events.append("close")

        original = _HARNESS._verify_pinned_shards
        _HARNESS._verify_pinned_shards = lambda _snapshot: (
            events.append("verify") or {"verified": {}}
        )
        self.addCleanup(lambda: setattr(_HARNESS, "_verify_pinned_shards", original))

        verified = _HARNESS._close_store_and_verify_pinned_shards(
            FakeStore(),
            self.root,
        )

        self.assertEqual(events, ["close", "verify"])
        self.assertEqual(verified, {"verified": {}})

    def test_hotset_is_revision_bound_and_leaves_dynamic_slot(self) -> None:
        hotset_path = self.root / "hotset.json"
        hotset_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": _HARNESS.PINNED_MODEL_ID,
                    "revision": _HARNESS.PINNED_REVISION,
                    "hotsets": {"0": [3, 1], "1": [2, 0]},
                }
            ),
            encoding="utf-8",
        )

        hotsets, digest = _HARNESS._load_hotsets(
            hotset_path,
            layers=(0, 1),
            experts_per_layer=4,
            slots_per_layer=3,
        )

        self.assertEqual(hotsets, {0: (3, 1), 1: (2, 0)})
        self.assertEqual(digest, hashlib.sha256(hotset_path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "dynamic LRU slot"):
            _HARNESS._load_hotsets(
                hotset_path,
                layers=(0, 1),
                experts_per_layer=4,
                slots_per_layer=2,
            )

    def test_reference_metadata_uses_existing_nested_baseline_shape(self) -> None:
        prompt = "Write a concise Python function."
        reference_path = self.root / "reference.json"
        reference_path.write_text(
            json.dumps(self._reference_payload(prompt)),
            encoding="utf-8",
        )

        loaded = _HARNESS._load_reference_metadata(
            reference_path,
            workload_id="python_code",
            prompt=prompt,
            max_new_tokens=2,
        )

        self.assertEqual(loaded["generated_token_ids"], [187, 187])
        self.assertEqual(loaded["source_generated_token_count"], 3)
        payload = self._reference_payload(prompt)
        payload["generation"]["temperature"] = 0.7
        reference_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "greedy temperature 0"):
            _HARNESS._load_reference_metadata(
                reference_path,
                workload_id="python_code",
                prompt=prompt,
                max_new_tokens=2,
            )

    def test_metric_delta_uses_counter_differences_and_invariants(self) -> None:
        before = SimpleNamespace(
            requests=10,
            hits=4,
            misses=6,
            evictions=2,
            storage_bytes=72,
            host_to_device_bytes=72,
            storage_loads=6,
            transfer_loads=6,
            adaptive_async_forwards=1,
            adaptive_sync_forwards=2,
            adaptive_async_experts=4,
            adaptive_sync_experts=5,
            pending_loads_peak=1,
            peak_staging_in_use=1,
            storage_seconds=1.0,
            transfer_seconds=2.0,
            forward_seconds=4.0,
        )
        after = SimpleNamespace(
            requests=14,
            hits=7,
            misses=7,
            evictions=3,
            storage_bytes=84,
            host_to_device_bytes=84,
            storage_loads=7,
            transfer_loads=7,
            adaptive_async_forwards=2,
            adaptive_sync_forwards=4,
            adaptive_async_experts=7,
            adaptive_sync_experts=9,
            pending_loads_peak=2,
            peak_staging_in_use=2,
            storage_seconds=1.5,
            transfer_seconds=2.25,
            forward_seconds=5.0,
        )

        delta = _HARNESS._metrics_delta(after, before)
        _HARNESS._validate_metric_delta(delta, expert_bytes=12)

        self.assertEqual(delta["requests"], 4)
        self.assertEqual(delta["hits"], 3)
        self.assertEqual(delta["misses"], 1)
        self.assertEqual(delta["hit_rate"], 0.75)
        self.assertEqual(delta["adaptive_async_forwards"], 1)
        self.assertEqual(delta["adaptive_sync_forwards"], 2)
        self.assertEqual(delta["adaptive_async_experts"], 3)
        self.assertEqual(delta["adaptive_sync_experts"], 4)
        self.assertNotIn("pending_loads_peak", delta)
        self.assertNotIn("peak_staging_in_use", delta)
        delta["storage_bytes"] = 13
        with self.assertRaisesRegex(RuntimeError, "storage bytes"):
            _HARNESS._validate_metric_delta(delta, expert_bytes=12)

        coalesced = {
            "requests": 2,
            "hits": 0,
            "misses": 2,
            "evictions": 0,
            "storage_bytes": 12,
            "host_to_device_bytes": 12,
            "storage_loads": 1,
            "transfer_loads": 1,
            "coalesced_requests": 1,
            "storage_seconds": 0.1,
            "transfer_seconds": 0.1,
            "forward_seconds": 0.1,
        }
        _HARNESS._validate_metric_delta(coalesced, expert_bytes=12)

    def test_percentile_ratio_and_create_only_json(self) -> None:
        self.assertIsNone(_HARNESS._ratio(1.0, 0.0))
        self.assertIsNone(_HARNESS._percentile([], 0.95))
        self.assertEqual(_HARNESS._percentile([1.0, 3.0], 0.50), 2.0)
        output = self.root / "nested" / "result.json"

        _HARNESS._write_json_create_only(output, {"status": "ok"})

        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "ok")
        with self.assertRaises(FileExistsError):
            _HARNESS._write_json_create_only(output, {"status": "replacement"})

    def test_cuda_overlap_aggregation_keeps_model_call_origins_separate(self) -> None:
        def capture(
            *,
            status: str,
            transfer_count: int,
            transfer_ms: float,
            compute_count: int,
            compute_ms: float,
            overlap_ms: float,
            reason: str | None = None,
        ) -> dict[str, object]:
            return {
                "complete": True,
                "status": status,
                "reason": reason,
                "summary": {
                    "transfer": {
                        "interval_count": transfer_count,
                        "active_duration_ms": transfer_ms,
                    },
                    "compute": {
                        "interval_count": compute_count,
                        "active_duration_ms": compute_ms,
                    },
                    "overlap": {
                        "duration_ms": overlap_ms,
                        "active_duration_saved_by_overlap_ms": overlap_ms,
                    },
                },
            }

        result = _HARNESS._aggregate_cuda_overlap(
            [
                capture(
                    status="measured",
                    transfer_count=2,
                    transfer_ms=4.0,
                    compute_count=2,
                    compute_ms=6.0,
                    overlap_ms=2.0,
                ),
                capture(
                    status="not_applicable",
                    transfer_count=0,
                    transfer_ms=0.0,
                    compute_count=1,
                    compute_ms=3.0,
                    overlap_ms=0.0,
                    reason="no expert H2D intervals were observed in this model call",
                ),
            ]
        )

        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["model_call_count"], 2)
        self.assertEqual(result["measured_model_call_count"], 1)
        self.assertEqual(result["h2d_interval_count"], 2)
        self.assertEqual(result["expert_compute_interval_count"], 3)
        self.assertEqual(result["h2d_union_ms"], 4.0)
        self.assertEqual(result["expert_compute_union_ms"], 9.0)
        self.assertEqual(result["overlap_ms"], 2.0)
        self.assertEqual(result["h2d_overlap_fraction"], 0.5)
        self.assertEqual(result["h2d_exposed_ms"], 2.0)

        unavailable = _HARNESS._aggregate_cuda_overlap(
            [
                capture(
                    status="not_applicable",
                    transfer_count=0,
                    transfer_ms=0.0,
                    compute_count=2,
                    compute_ms=4.0,
                    overlap_ms=0.0,
                    reason="no expert H2D intervals were observed in this model call",
                )
            ]
        )
        self.assertEqual(unavailable["status"], "not_applicable")
        self.assertIn("no expert H2D", unavailable["reason"])

        long_capture_durations = [1000.0, *([0.1] * 64)]
        long_result = _HARNESS._aggregate_cuda_overlap(
            [
                capture(
                    status="measured",
                    transfer_count=1,
                    transfer_ms=duration,
                    compute_count=1,
                    compute_ms=duration,
                    overlap_ms=duration,
                )
                for duration in long_capture_durations
            ]
        )
        self.assertEqual(
            long_result["h2d_union_ms"],
            math.fsum(long_capture_durations),
        )

    def test_oom_classifier_is_narrow(self) -> None:
        self.assertTrue(_HARNESS._is_cuda_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(_HARNESS._is_cuda_oom(RuntimeError("CPU out of memory")))
        self.assertFalse(_HARNESS._is_cuda_oom(ValueError("bad input")))

    def test_source_provenance_binds_commit_and_script_hash(self) -> None:
        source = _HARNESS._source_provenance()
        demo_source = _HARNESS._source_provenance(best_effort=True)

        self.assertRegex(str(source["commit"]), r"^[0-9a-f]{40}$")
        self.assertIsInstance(source["tree_clean"], bool)
        self.assertEqual(source["provenance_mode"], "benchmark_evidence")
        self.assertEqual(source["benchmark_script"], "scripts/benchmark_paged_olmoe.py")
        self.assertEqual(
            source["benchmark_script_sha256"],
            hashlib.sha256(_SCRIPT.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            source["paged_runtime_sha256"],
            hashlib.sha256(
                (_SCRIPT.parents[1] / "src" / "moevm" / "paged_runtime.py").read_bytes()
            ).hexdigest(),
        )
        self.assertRegex(str(demo_source["commit"]), r"^[0-9a-f]{40}$")
        self.assertIsNone(demo_source["tree_clean"])
        self.assertIsInstance(demo_source["git_tree_clean_observed"], bool)
        self.assertEqual(demo_source["provenance_mode"], "demo")

    def test_demo_provenance_is_best_effort_without_git(self) -> None:
        with mock.patch.object(
            _HARNESS.subprocess,
            "run",
            side_effect=FileNotFoundError("git is unavailable"),
        ):
            source = _HARNESS._source_provenance(best_effort=True)
            with self.assertRaisesRegex(RuntimeError, "Git provenance is required"):
                _HARNESS._source_provenance()

        self.assertIsNone(source["commit"])
        self.assertIsNone(source["tree_clean"])
        self.assertFalse(source["git_available"])
        self.assertEqual(source["provenance_mode"], "demo")
        self.assertEqual(
            source["benchmark_script"],
            _SCRIPT.relative_to(_SCRIPT.parents[1]).as_posix(),
        )

    @unittest.skipUnless(torch is not None, "requires torch for timeline payload")
    def test_inference_pass_attaches_opt_in_cuda_overlap_payload(self) -> None:
        zero_metrics = SimpleNamespace(
            requests=0,
            hits=0,
            misses=0,
            evictions=0,
            storage_bytes=0,
            host_to_device_bytes=0,
            storage_seconds=0.0,
            transfer_seconds=0.0,
            forward_seconds=0.0,
        )
        call_timeline = {
            "complete": True,
            "status": "measured",
            "method": "cuda_events_v1",
            "scope": "paged_expert_h2d_vs_expert_compute",
            "unit": "milliseconds",
            "reason": None,
            "summary": {
                "transfer": {"interval_count": 1, "active_duration_ms": 4.0},
                "compute": {"interval_count": 1, "active_duration_ms": 5.0},
                "overlap": {
                    "duration_ms": 2.0,
                    "active_duration_saved_by_overlap_ms": 2.0,
                },
            },
        }

        class Capture:
            result = call_timeline

            def __enter__(self):
                return self

            def __exit__(
                self, _exc_type: object, _exc_value: object, _traceback: object
            ) -> bool:
                return False

        class FakeRuntime:
            def metrics(self):
                return zero_metrics

            @staticmethod
            def cuda_timeline_capture():
                return Capture()

        class FakeTokenizer:
            eos_token_id = None

            @staticmethod
            def decode(token_ids, **_kwargs):
                return ",".join(str(token) for token in token_ids)

        class FakeModel:
            @staticmethod
            def __call__(**_kwargs):
                logits = torch.zeros(1, 1, 8)
                logits[..., 3] = 1.0
                return SimpleNamespace(logits=logits, past_key_values={})

        originals = {
            name: getattr(_HARNESS, name)
            for name in (
                "_sync_cuda",
                "_reset_cuda_peak",
                "_cuda_memory_payload",
                "_process_memory",
            )
        }
        _HARNESS._sync_cuda = lambda _torch: None
        _HARNESS._reset_cuda_peak = lambda _torch: 0
        _HARNESS._cuda_memory_payload = lambda _torch, _baseline: {}
        _HARNESS._process_memory = lambda: {
            "rss_bytes": 0,
            "peak_rss_bytes": 0,
        }
        self.addCleanup(
            lambda: [
                setattr(_HARNESS, name, value) for name, value in originals.items()
            ]
        )

        result = _HARNESS._run_inference_pass(
            label="cold_expert_cache",
            torch=torch,
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            runtime=FakeRuntime(),
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            max_new_tokens=1,
            expert_bytes=12,
            capture_cuda_overlap=True,
        )

        self.assertEqual(result["cuda_overlap"]["status"], "measured")
        self.assertEqual(result["cuda_overlap"]["h2d_union_ms"], 4.0)
        self.assertEqual(result["cuda_overlap"]["overlap_ms"], 2.0)
        self.assertEqual(result["prefill"]["cuda_event_timeline"], call_timeline)
        self.assertEqual(result["first_token"]["cuda_event_timeline"], call_timeline)

    @unittest.skipUnless(torch is not None, "requires torch for manual decode contract")
    def test_manual_two_token_decode_uses_fresh_kv_and_extended_mask(self) -> None:
        zero_metrics = SimpleNamespace(
            requests=0,
            hits=0,
            misses=0,
            evictions=0,
            storage_bytes=0,
            host_to_device_bytes=0,
            storage_seconds=0.0,
            transfer_seconds=0.0,
            forward_seconds=0.0,
        )

        class FakeRuntime:
            def metrics(self):
                return zero_metrics

        class FakeTokenizer:
            eos_token_id = None

            @staticmethod
            def decode(token_ids, **_kwargs):
                return ",".join(str(token) for token in token_ids)

        class FakeModel:
            def __init__(self) -> None:
                self.calls = []

            def __call__(
                self,
                *,
                input_ids,
                attention_mask,
                past_key_values=None,
                **_kwargs,
            ):
                self.calls.append(
                    {
                        "input_shape": tuple(input_ids.shape),
                        "input_ids": tuple(int(value) for value in input_ids.flatten()),
                        "mask_shape": tuple(attention_mask.shape),
                        "past": past_key_values,
                    }
                )
                expected_token = 4 if past_key_values is None else 6
                logits = torch.zeros(1, 1, 8)
                logits[..., expected_token] = 1.0
                return SimpleNamespace(
                    logits=logits,
                    past_key_values={"length": attention_mask.shape[-1]},
                )

        originals = {
            name: getattr(_HARNESS, name)
            for name in (
                "_sync_cuda",
                "_reset_cuda_peak",
                "_cuda_memory_payload",
                "_process_memory",
            )
        }
        _HARNESS._sync_cuda = lambda _torch: None
        _HARNESS._reset_cuda_peak = lambda _torch: 0
        _HARNESS._cuda_memory_payload = lambda _torch, _baseline: {}
        _HARNESS._process_memory = lambda: {
            "rss_bytes": 0,
            "peak_rss_bytes": 0,
        }
        self.addCleanup(
            lambda: [
                setattr(_HARNESS, name, value) for name, value in originals.items()
            ]
        )
        model = FakeModel()

        result = _HARNESS._run_inference_pass(
            label="cold_expert_cache",
            torch=torch,
            model=model,
            tokenizer=FakeTokenizer(),
            runtime=FakeRuntime(),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=2,
            expert_bytes=12,
        )

        self.assertEqual(result["generated_ids"], [4, 6])
        self.assertEqual(result["decode"]["token_count"], 1)
        self.assertEqual(model.calls[0]["input_shape"], (1, 3))
        self.assertIsNone(model.calls[0]["past"])
        self.assertEqual(model.calls[1]["input_shape"], (1, 1))
        self.assertEqual(model.calls[1]["mask_shape"], (1, 4))
        self.assertEqual(model.calls[1]["past"], {"length": 3})

        teacher_forced_model = FakeModel()
        FakeTokenizer.eos_token_id = 5
        teacher_forced = _HARNESS._run_inference_pass(
            label="cold_expert_cache",
            torch=torch,
            model=teacher_forced_model,
            tokenizer=FakeTokenizer(),
            runtime=FakeRuntime(),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=2,
            expert_bytes=12,
            forced_token_ids=[5, 7],
        )

        self.assertEqual(teacher_forced["generated_ids"], [4, 6])
        self.assertEqual(teacher_forced["fed_token_ids"], [5, 7])
        self.assertTrue(teacher_forced["teacher_forced"])
        self.assertEqual(
            teacher_forced["reference_prediction"],
            {
                "matched_tokens": 0,
                "total_tokens": 2,
                "match_rate": 0.0,
                "exact_match": False,
                "first_mismatch_index": 0,
            },
        )
        self.assertEqual(teacher_forced_model.calls[1]["input_ids"], (5,))
        self.assertEqual(teacher_forced["decode"]["token_count"], 1)


if __name__ == "__main__":
    unittest.main()
