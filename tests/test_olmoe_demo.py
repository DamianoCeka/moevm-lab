from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "olmoe_demo.py"
_SPEC = importlib.util.spec_from_file_location("olmoe_demo", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = demo
_SPEC.loader.exec_module(demo)

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REVISION = "bd1c52f59153f724c1ad11ca1791edc77bab3806"
SHARDS = {"model-00001-of-00003.safetensors": "a" * 64}


def _metrics(*, hits: int = 2, misses: int = 3) -> dict[str, int | float]:
    loads = misses
    return {
        "requests": hits + misses,
        "hits": hits,
        "misses": misses,
        "evictions": 1,
        "storage_loads": loads,
        "transfer_loads": loads,
        "storage_bytes": loads * demo.EXPERT_BYTES,
        "host_to_device_bytes": loads * demo.EXPERT_BYTES,
        "coalesced_requests": 0,
        "admission_rejections": 0,
        "storage_failures": 0,
        "transfer_failures": 0,
        "staging_waits": 0,
        "storage_seconds": 0.1,
        "transfer_seconds": 0.1,
        "forward_seconds": 0.1,
        "storage_queue_seconds": 0.0,
        "demand_wait_seconds": 0.0,
        "hit_rate": hits / (hits + misses),
    }


def _cuda(peak: int) -> dict[str, int]:
    return {
        "baseline_allocated_bytes": 100,
        "peak_allocated_bytes": peak,
        "peak_incremental_bytes": peak - 100,
        "peak_reserved_bytes": peak + 1024,
    }


def _process(rss: int) -> dict[str, int]:
    return {"rss_bytes": rss, "peak_rss_bytes": rss + 4096}


def _result(
    mode: str,
    *,
    slots: int,
    script_sha256: str,
    prompt_sha256: str,
    wall_scale: float = 1.0,
) -> dict[str, object]:
    def pass_payload(name: str, wall: float, peak: int, rss: int, misses: int):
        return {
            "label": name,
            "total_wall_seconds": wall,
            "generated_token_count": 2,
            "end_to_end_generated_tokens_per_second_including_prefill": 2 / wall,
            "generated_ids": [11, 12],
            "fed_token_ids": [11, 12],
            "generated_text": " demo",
            "metrics": _metrics(hits=2, misses=misses),
            "cuda_memory": _cuda(peak),
            "process_memory_after": _process(rss),
        }

    cache_bytes = slots * demo.SLOT_FOOTPRINT_BYTES
    return {
        "schema_version": 1,
        "status": "ok",
        "evidence": {
            "label": "interactive local hardware demo; not benchmark evidence",
            "publishable_benchmark_evidence": False,
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "shards": {
                filename: {"sha256": digest, "size_bytes": 1}
                for filename, digest in SHARDS.items()
            },
        },
        "runtime": {
            "device": "cuda:0",
            "device_uuid": "fake-uuid",
            "device_name": "Fake GPU",
            "policy": "lru",
            "pipeline": mode,
            "budget": {
                "slots_per_layer": slots,
                "staging_slots": 2,
                "cache_bytes": cache_bytes,
                "non_expert_checkpoint_bytes": 953_421_824,
                "device_total_vram_bytes": 12_288 * 1024**2,
            },
        },
        "workload": {
            "id": "systems_it",
            "input_tokens": 21,
            "max_new_tokens": 2,
            "prompt_sha256": prompt_sha256,
            "seed": 17,
        },
        "model_load": {
            "total_seconds": 0.75 * wall_scale,
            "cuda_memory": _cuda(6_000),
            "process_memory_after": _process(7_000),
        },
        "passes": {
            "cold_expert_cache": pass_payload(
                "cold_expert_cache", 4.0 * wall_scale, 10_000, 20_000, 3
            ),
            "repeat_retained_expert_cache": pass_payload(
                "repeat_retained_expert_cache",
                2.0 * wall_scale,
                11_000,
                22_000,
                2,
            ),
        },
        "source": {
            "benchmark_script_sha256": script_sha256,
            "paged_runtime_sha256": demo._sha256(demo.PAGED_RUNTIME_SOURCE),
            "commit": "1" * 40,
            "tree_clean": None,
            "git_available": True,
            "git_tree_clean_observed": False,
            "provenance_mode": "demo",
        },
    }


class FakeRunner:
    def __init__(self, *, benchmark_scale: dict[str, float] | None = None):
        self.calls: list[tuple[list[str], Path, dict[str, str] | None]] = []
        self.benchmark_scale = benchmark_scale or {"async": 1.0, "sync": 1.25}

    def run(self, argv, *, cwd, env=None):
        arguments = list(argv)
        copied_env = None if env is None else dict(env)
        self.calls.append((arguments, cwd, copied_env))
        if arguments[0] == "nvidia-smi":
            return demo.ProcessResult(
                0,
                "0, GPU-fake-uuid, Fake GPU, 12288, 11000, 8.6, 610.47\n"
                "1, GPU-other, Other GPU, 8192, 4096, 8.0, 610.47\n",
                "",
            )
        mode = arguments[arguments.index("--pipeline") + 1]
        output = Path(arguments[arguments.index("--output") + 1])
        slots = int(arguments[arguments.index("--slots-per-layer") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        script = Path(arguments[1])
        prompt_sha256 = demo.hashlib.sha256(b"Spiega MoEVM.").hexdigest()
        output.write_text(
            json.dumps(
                _result(
                    mode,
                    slots=slots,
                    script_sha256=demo._sha256(script),
                    prompt_sha256=prompt_sha256,
                    wall_scale=self.benchmark_scale[mode],
                )
            ),
            encoding="utf-8",
        )
        return demo.ProcessResult(0, f"Wrote {output}\n", "")


class DemoFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot with spaces"
        self.snapshot.mkdir()
        total = (
            demo.LAYER_COUNT * demo.EXPERTS_PER_LAYER * demo.EXPERT_BYTES + 953_421_824
        )
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": {}}),
            encoding="utf-8",
        )
        self.workload = self.root / "workloads with spaces.json"
        self.workload.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workloads": [
                        {
                            "id": "systems_it",
                            "category": "systems",
                            "language": "it",
                            "prompt": "Spiega MoEVM.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.python = self.root / "python fake.exe"
        self.python.write_text("", encoding="utf-8")
        self.benchmark = self.root / "benchmark fake.py"
        self.benchmark.write_text("# fixture\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **overrides):
        values = {
            "snapshot": self.snapshot,
            "output_root": self.root / "output with spaces",
            "workload_file": self.workload,
            "workload_id": "systems_it",
            "device": "cuda:0",
            "requested_slots": "auto",
            "max_input_tokens": 64,
            "max_new_tokens": 2,
            "compare": False,
            "dry_run": False,
            "python_executable": self.python,
            "benchmark_script": self.benchmark,
        }
        values.update(overrides)
        return demo.DemoConfig(**values)

    def execute(self, config, runner):
        return demo.execute_demo(
            config,
            model_id=MODEL_ID,
            revision=REVISION,
            shard_sha256=SHARDS,
            runner=runner,
        )


class PlannerTests(DemoFixture):
    def test_auto_slots_use_guarded_nvidia_smi_budget(self):
        gpu = demo.GpuInfo(
            index=0,
            uuid="fake-uuid",
            name="GPU",
            total_bytes=12 * 1024**3,
            free_bytes=10 * 1024**3,
            compute_capability="8.6",
            driver_version="610.47",
        )
        selected, reserve, affordable = demo.plan_slots(gpu, 953_421_824, "auto")
        self.assertEqual(reserve, int(12 * 1024**3 * 0.20))
        self.assertGreaterEqual(affordable, 32)
        self.assertEqual(selected, 32)

    def test_explicit_slots_above_budget_fail_closed(self):
        gpu = demo.GpuInfo(
            index=0,
            uuid="fake-uuid",
            name="GPU",
            total_bytes=8 * 1024**3,
            free_bytes=4 * 1024**3,
            compute_capability="8.6",
            driver_version="610.47",
        )
        with self.assertRaisesRegex(demo.DemoError, "only .* fit"):
            demo.plan_slots(gpu, 953_421_824, "32")

    def test_explicit_one_slot_is_supported_when_it_fits(self):
        gpu = demo.GpuInfo(
            index=0,
            uuid="fake-uuid",
            name="GPU",
            total_bytes=8 * 1024**3,
            free_bytes=4 * 1024**3,
            compute_capability="8.6",
            driver_version="610.47",
        )
        selected, _reserve, affordable = demo.plan_slots(gpu, 953_421_824, "1")
        self.assertGreaterEqual(affordable, 1)
        self.assertEqual(selected, 1)

    def test_dry_run_does_not_write_or_run_benchmark(self):
        runner = FakeRunner()
        plan, run_dir = self.execute(self.config(dry_run=True), runner)
        self.assertEqual(plan["runtime"]["pipeline_modes"], ["async"])
        self.assertFalse(run_dir.exists())
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0][0][0], "nvidia-smi")


class ExecutionTests(DemoFixture):
    def test_validation_accepts_driver_reserved_cuda_capacity(self):
        class DriverReservedCapacityRunner(FakeRunner):
            def run(self, argv, *, cwd, env=None):
                result = super().run(argv, cwd=cwd, env=env)
                arguments = list(argv)
                if "--pipeline" in arguments:
                    output = Path(arguments[arguments.index("--output") + 1])
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    payload["runtime"]["budget"]["device_total_vram_bytes"] = int(
                        12_288 * 1024**2 * 0.987
                    )
                    output.write_text(json.dumps(payload), encoding="utf-8")
                return result

        summary, _path = self.execute(self.config(), DriverReservedCapacityRunner())
        self.assertEqual(summary["status"], "ok")

    def test_validation_rejects_large_cuda_capacity_mismatch(self):
        class WrongCapacityRunner(FakeRunner):
            def run(self, argv, *, cwd, env=None):
                result = super().run(argv, cwd=cwd, env=env)
                arguments = list(argv)
                if "--pipeline" in arguments:
                    output = Path(arguments[arguments.index("--output") + 1])
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    payload["runtime"]["budget"]["device_total_vram_bytes"] = int(
                        12_288 * 1024**2 * 0.95
                    )
                    output.write_text(json.dumps(payload), encoding="utf-8")
                return result

        with self.assertRaisesRegex(demo.DemoError, "CUDA capacity does not match"):
            self.execute(self.config(), WrongCapacityRunner())

    def test_default_async_uses_argv_list_offline_and_paths_with_spaces(self):
        runner = FakeRunner()
        summary, summary_path = self.execute(self.config(), runner)
        benchmark_calls = [call for call in runner.calls if "--pipeline" in call[0]]
        self.assertEqual(len(benchmark_calls), 1)
        argv, _cwd, env = benchmark_calls[0]
        self.assertIn("--demo-mode", argv)
        self.assertEqual(argv[argv.index("--pipeline") + 1], "async")
        self.assertEqual(argv[argv.index("--staging-slots") + 1], "2")
        self.assertEqual(argv[argv.index("--snapshot") + 1], str(self.snapshot))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertTrue(summary_path.is_file())
        self.assertEqual(summary["plan"]["runtime"]["pipeline_modes"], ["async"])

    def test_compare_enforces_identity_and_reports_savings(self):
        runner = FakeRunner(benchmark_scale={"sync": 1.25, "async": 1.0})
        summary, _path = self.execute(self.config(compare=True), runner)
        modes = [
            call[0][call[0].index("--pipeline") + 1]
            for call in runner.calls
            if "--pipeline" in call[0]
        ]
        self.assertEqual(modes, ["sync", "async"])
        comparison = summary["comparison"]["cold_expert_cache"]
        self.assertTrue(math.isclose(comparison["sync_over_async_ratio"], 1.25))
        self.assertTrue(math.isclose(comparison["saving_fraction"], 0.20))

    def test_compare_rejects_counter_mismatch(self):
        class MismatchRunner(FakeRunner):
            def run(self, argv, *, cwd, env=None):
                result = super().run(argv, cwd=cwd, env=env)
                arguments = list(argv)
                if "--pipeline" in arguments:
                    mode = arguments[arguments.index("--pipeline") + 1]
                    if mode == "async":
                        output = Path(arguments[arguments.index("--output") + 1])
                        payload = json.loads(output.read_text(encoding="utf-8"))
                        metrics = payload["passes"]["cold_expert_cache"]["metrics"]
                        metrics["hits"] += 1
                        metrics["requests"] += 1
                        output.write_text(json.dumps(payload), encoding="utf-8")
                return result

        with self.assertRaisesRegex(demo.DemoError, "identity gate"):
            self.execute(self.config(compare=True), MismatchRunner())

    def test_resume_reuses_valid_results_and_summary_create_only(self):
        first = FakeRunner()
        summary, summary_path = self.execute(self.config(), first)
        original = summary_path.read_bytes()
        second = FakeRunner()
        resumed, resumed_path = self.execute(self.config(), second)
        self.assertEqual(resumed, summary)
        self.assertEqual(resumed_path.read_bytes(), original)
        self.assertFalse(any("--pipeline" in call[0] for call in second.calls))

    def test_invalid_existing_result_is_not_overwritten(self):
        runner = FakeRunner()
        _summary, summary_path = self.execute(self.config(), runner)
        async_path = summary_path.parent / "async.json"
        async_path.write_text("{broken", encoding="utf-8")
        before = async_path.read_bytes()
        with self.assertRaisesRegex(demo.DemoError, "valid async benchmark JSON"):
            self.execute(self.config(), FakeRunner())
        self.assertEqual(async_path.read_bytes(), before)

    def test_summary_contains_memory_speed_budget_and_no_absolute_paths(self):
        summary, _path = self.execute(self.config(), FakeRunner())
        self.assertFalse(summary["publishable_benchmark_evidence"])
        mode = summary["modes"]["async"]
        self.assertEqual(mode["model_load_seconds"], 0.75)
        self.assertEqual(mode["passes"]["cold_expert_cache"]["tokens_per_second"], 0.5)
        self.assertEqual(
            mode["passes"]["cold_expert_cache"]["generated_token_ids"], [11, 12]
        )
        self.assertEqual(mode["memory"]["peak_allocated_vram_bytes"], 11_000)
        self.assertEqual(mode["memory"]["peak_reserved_vram_bytes"], 12_024)
        self.assertEqual(mode["memory"]["peak_process_working_set_bytes"], 26_096)
        self.assertEqual(
            summary["cache_budget"]["staging_host_bytes"], 2 * 12 * 1024**2
        )
        encoded = json.dumps(summary)
        self.assertNotIn(str(self.root), encoded)
        self.assertGreaterEqual(len(summary["limitations"]), 5)

    def test_throughput_inconsistency_is_rejected(self):
        runner = FakeRunner()
        summary, summary_path = self.execute(self.config(), runner)
        result_path = summary_path.parent / "async.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["passes"]["cold_expert_cache"][
            "end_to_end_generated_tokens_per_second_including_prefill"
        ] = 999.0
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(demo.DemoError, "throughput is inconsistent"):
            self.execute(self.config(), FakeRunner())
        self.assertEqual(summary["status"], "ok")


if __name__ == "__main__":
    unittest.main()
