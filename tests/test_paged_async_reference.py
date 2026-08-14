from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import statistics
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_DIR = (
    _ROOT / "benchmarks" / "reference" / "paged-runtime-olmoe-p310-async-smoke"
)
_REFERENCE = _REFERENCE_DIR / "result.json"
_SVG = _REFERENCE_DIR / "sync-vs-async.svg"
_RENDERER = _ROOT / "scripts" / "render_paged_async_reference.py"
_SPEC = importlib.util.spec_from_file_location(
    "render_paged_async_reference", _RENDERER
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import async reference renderer: {_RENDERER}")
_CHART = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHART)


class PagedAsyncReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = _REFERENCE.read_text(encoding="utf-8")
        cls.reference = json.loads(cls.raw)

    def test_reference_is_sanitized_and_bound_to_clean_source(self) -> None:
        self.assertNotRegex(self.raw, re.compile(r"[A-Za-z]:[/\\]"))
        self.assertNotIn("Users", self.raw)
        self.assertNotIn('"hostname"', self.raw)

        source = self.reference["source"]
        self.assertTrue(source["tree_clean"])
        self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(source["benchmark_script_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(source["pair_comparator_sha256"], r"^[0-9a-f]{64}$")

        protocol = self.reference["protocol"]
        self.assertEqual(protocol["paired_repetitions"], 3)
        self.assertEqual(
            protocol["alternating_order"],
            ["async-sync", "sync-async", "async-sync"],
        )
        self.assertEqual(protocol["slots_per_layer"], 32)
        self.assertEqual(protocol["staging_slots"], 2)

    def test_pair_statistics_and_primitives_are_recomputed(self) -> None:
        correctness = self.reference["correctness"]
        for field in (
            "all_pair_gates_passed",
            "cross_repetition_identity_exact",
            "generated_and_fed_token_ids_exact",
            "cache_and_transfer_primitives_exact",
            "all_admission_rejections_zero",
            "all_storage_failures_zero",
            "all_transfer_failures_zero",
            "all_coalesced_requests_zero",
            "peak_allocated_vram_equal_between_modes",
        ):
            self.assertTrue(correctness[field], field)
        self.assertEqual(correctness["pair_gates_passed"], 3)
        self.assertEqual(correctness["generated_token_ids"], [187, 187])
        self.assertEqual(correctness["fed_token_ids"], [187, 187])
        self.assertEqual(
            correctness["generated_token_ids"],
            self.reference["workload"]["reference_token_ids"],
        )

        expert_bytes = self.reference["model"]["expert_bytes"]
        for condition in self.reference["conditions"].values():
            ratios: list[float] = []
            savings: list[float] = []
            fractions: list[float] = []
            sync_times: list[float] = []
            async_times: list[float] = []
            for row in condition["repetitions"]:
                sync_time = row["sync_wall_seconds"]
                async_time = row["async_wall_seconds"]
                ratio = sync_time / async_time
                saving = sync_time - async_time
                fraction = saving / sync_time
                self.assertGreater(sync_time, async_time)
                self.assertTrue(math.isclose(row["sync_over_async_ratio"], ratio))
                self.assertTrue(math.isclose(row["saving_seconds"], saving))
                self.assertTrue(math.isclose(row["saving_fraction"], fraction))
                ratios.append(ratio)
                savings.append(saving)
                fractions.append(fraction)
                sync_times.append(sync_time)
                async_times.append(async_time)

            aggregate = condition["aggregate"]
            ratio_median = statistics.median(ratios)
            ratio_mad = statistics.median(abs(value - ratio_median) for value in ratios)
            expected = {
                "median_sync_wall_seconds": statistics.median(sync_times),
                "median_async_wall_seconds": statistics.median(async_times),
                "paired_ratio_median": ratio_median,
                "paired_ratio_min": min(ratios),
                "paired_ratio_max": max(ratios),
                "paired_ratio_mad": ratio_mad,
                "paired_ratio_mad_over_median": ratio_mad / ratio_median,
                "paired_saving_seconds_median": statistics.median(savings),
                "paired_time_saved_fraction_median": statistics.median(fractions),
            }
            for field, value in expected.items():
                self.assertTrue(math.isclose(aggregate[field], value), field)
            self.assertTrue(aggregate["all_repetitions_faster"])

            primitives = condition["identical_primitives_per_mode"]
            self.assertEqual(
                primitives["requests"], primitives["hits"] + primitives["misses"]
            )
            self.assertEqual(primitives["storage_loads"], primitives["misses"])
            self.assertEqual(primitives["transfer_loads"], primitives["misses"])
            self.assertEqual(
                primitives["storage_bytes"], primitives["misses"] * expert_bytes
            )
            self.assertEqual(
                primitives["host_to_device_bytes"], primitives["storage_bytes"]
            )

    def test_raw_hashes_are_relative_and_match_when_present(self) -> None:
        for relative_path, expected_digest in self.reference[
            "raw_artifacts_sha256"
        ].items():
            self.assertFalse(Path(relative_path).is_absolute())
            self.assertRegex(expected_digest, r"^[0-9a-f]{64}$")
            raw_path = _ROOT / relative_path
            if raw_path.is_file():
                actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                self.assertEqual(actual, expected_digest)

    def test_svg_is_deterministic_accessible_and_static(self) -> None:
        _CHART.validate_reference(self.reference)
        expected = _CHART.render_svg(self.reference)
        actual = _SVG.read_text(encoding="utf-8")
        self.assertEqual(actual, expected)
        for required in (
            '<title id="chart-title">',
            '<desc id="chart-desc">',
            'role="img"',
            "lower is better",
            "n=3 pairs",
            "does not prove physical NVMe/CUDA interval overlap",
        ):
            self.assertIn(required, actual)
        for forbidden in ("<script", "<foreignObject", "<image", "http://", "https://"):
            if forbidden == "http://":
                self.assertEqual(actual.count(forbidden), 1)
            else:
                self.assertNotIn(forbidden, actual)


if __name__ == "__main__":
    unittest.main()
