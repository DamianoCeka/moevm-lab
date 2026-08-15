from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "benchmarks" / "reference" / "paged-runtime-olmoe-runpod-rtx6000ada-study"
)
RESULT_PATH = REFERENCE / "result.json"
SVG_PATH = REFERENCE / "study.svg"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PagedRunpodReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        cls.summarizer = _load_module(
            "runpod_study_summarizer",
            ROOT / "scripts" / "summarize_paged_runpod_study.py",
        )
        cls.renderer = _load_module(
            "runpod_study_renderer",
            ROOT / "scripts" / "render_paged_runpod_study.py",
        )

    def test_reference_is_sanitized_and_complete(self) -> None:
        encoded = RESULT_PATH.read_text(encoding="utf-8")
        for private_fragment in (
            "C:\\\\",
            "C:/Users/",
            "/workspace/",
            "/root/",
            "njdr3o2ukhzupg",
        ):
            self.assertNotIn(private_fragment, encoded)

        self.assertEqual(self.payload["schema_version"], 1)
        self.assertEqual(len(self.payload["observations"]), 72)
        self.assertEqual(len(self.payload["condition_aggregates"]), 24)
        self.assertEqual(len(self.payload["raw_artifacts_sha256"]), 108)
        self.assertTrue(self.payload["correctness"]["all_pair_gates_passed"])
        for digest in self.payload["raw_artifacts_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_observation_equations_and_headline(self) -> None:
        faster = {"cold": 0, "retained": 0}
        for row in self.payload["observations"]:
            sync = row["sync_wall_seconds"]
            async_ = row["async_wall_seconds"]
            self.assertTrue(
                math.isclose(row["sync_over_async_ratio"], sync / async_, rel_tol=1e-12)
            )
            self.assertTrue(
                math.isclose(
                    row["saving_fraction"], (sync - async_) / sync, rel_tol=1e-12
                )
            )
            metrics = row["metrics"]
            self.assertEqual(metrics["requests"], metrics["hits"] + metrics["misses"])
            for field in (
                "admission_rejections",
                "coalesced_requests",
                "storage_failures",
                "transfer_failures",
            ):
                self.assertEqual(metrics[field], 0)
            faster[row["cache_condition"]] += row["sync_over_async_ratio"] > 1.0

        headline = self.payload["headline"]
        self.assertEqual(faster, {"cold": 33, "retained": 18})
        self.assertEqual(headline["cold_faster_comparisons"], 33)
        self.assertEqual(headline["retained_faster_comparisons"], 18)

    def test_aggregates_recompute(self) -> None:
        rows = self.payload["observations"]
        recomputed = self.summarizer._condition_aggregates(rows)
        self.assertEqual(recomputed, self.payload["condition_aggregates"])
        core = self.summarizer._core_aggregate(rows)
        self.assertEqual(core, self.payload["core_aggregate"])

        cold_ratios = [
            row["sync_over_async_ratio"] for row in core["cold"]["repetitions"]
        ]
        self.assertTrue(
            math.isclose(
                statistics.median(cold_ratios),
                self.payload["headline"]["core_cold_paired_ratio_median"],
                rel_tol=1e-12,
            )
        )

    def test_renderer_is_deterministic_static_and_accessible(self) -> None:
        rendered = self.renderer.render(self.payload)
        self.assertEqual(rendered, SVG_PATH.read_text(encoding="utf-8"))
        self.assertIn("<title", rendered)
        self.assertIn("<desc", rendered)
        self.assertIn('role="img"', rendered)
        self.assertIn("async faster", rendered)
        self.assertNotIn("<script", rendered.lower())
        self.assertNotIn("<image", rendered.lower())
        self.assertNotIn("foreignObject", rendered)
        self.assertEqual(
            hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            hashlib.sha256(SVG_PATH.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
