from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path

_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "reference"
    / "paged-runtime-olmoe-p310-multiworkload"
    / "result.json"
)


class PagedMultiworkloadReferenceTests(unittest.TestCase):
    def test_reference_is_sanitized_and_aggregate_is_derived(self) -> None:
        raw = _REFERENCE.read_text(encoding="utf-8")
        result = json.loads(raw)

        self.assertNotIn("C:\\", raw)
        self.assertNotIn("Users", raw)
        self.assertRegex(result["source_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(len(result["workloads"]), 5)

        workloads = result["workloads"]
        aggregate = result["aggregate"]
        baseline = sum(row["baseline_wall_seconds"] for row in workloads)
        cold = sum(row["paged_cold_wall_seconds"] for row in workloads)
        retained = sum(row["paged_retained_wall_seconds"] for row in workloads)
        cold_bytes = sum(row["paged_cold_logical_storage_bytes"] for row in workloads)
        retained_bytes = sum(
            row["paged_retained_logical_storage_bytes"] for row in workloads
        )

        self.assertTrue(result["correctness"]["all_runs_status_ok"])
        self.assertTrue(result["correctness"]["all_fed_reference_sequences_exact"])
        self.assertTrue(result["correctness"]["cold_retained_predictions_identical"])
        self.assertEqual(
            sum(row["prediction_matches"] for row in workloads),
            result["correctness"]["prediction_matches"],
        )
        self.assertEqual(result["correctness"]["prediction_matches"], 78)
        self.assertEqual(result["correctness"]["prediction_total"], 80)

        self.assertTrue(math.isclose(aggregate["baseline_wall_seconds"], baseline))
        self.assertTrue(math.isclose(aggregate["paged_cold_wall_seconds"], cold))
        self.assertTrue(
            math.isclose(aggregate["paged_retained_wall_seconds"], retained)
        )
        self.assertTrue(
            math.isclose(aggregate["cold_speedup_over_baseline"], baseline / cold)
        )
        self.assertTrue(
            math.isclose(
                aggregate["retained_speedup_over_baseline"], baseline / retained
            )
        )
        self.assertEqual(aggregate["cold_logical_storage_bytes"], cold_bytes)
        self.assertEqual(aggregate["retained_logical_storage_bytes"], retained_bytes)
        self.assertTrue(
            math.isclose(
                aggregate["retained_logical_traffic_reduction"],
                1 - retained_bytes / cold_bytes,
            )
        )

        for row in workloads:
            self.assertGreater(row["cold_speedup_over_baseline"], 1.0)
            self.assertGreater(row["retained_speedup_over_baseline"], 1.0)
            self.assertLess(
                row["paged_peak_vram_bytes"], row["baseline_peak_vram_bytes"]
            )
            self.assertRegex(row["raw_baseline_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["raw_paged_sha256"], r"^[0-9a-f]{64}$")

        self.assertEqual(
            {
                check["workload"]
                for check in result["correctness"]["differential_checks"]
            },
            {"systems_en", "math_reasoning"},
        )
        self.assertNotRegex(raw, re.compile(r"[A-Za-z]:[/\\]"))
