from __future__ import annotations

import hashlib
import json
import statistics
import unittest
from pathlib import Path

from moevm.config import load_config
from moevm.simulator import compare_experiment
from moevm.trace import read_trace


def _row(columns: list[str], values: list[object]) -> dict[str, object]:
    return dict(zip(columns, values, strict=True))


def _linear_fit(points: list[tuple[int, float]]) -> dict[str, float]:
    """Fit latency seconds against decimal GB using ordinary least squares."""
    x_values = [block_mib * 1024 * 1024 / 1_000_000_000 for block_mib, _ in points]
    y_values = [latency_ms / 1000 for _, latency_ms in points]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * value for value in x_values]
    residual = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(y_values, predicted, strict=True)
    )
    total = sum((value - y_mean) ** 2 for value in y_values)
    return {
        "bandwidth_gbps": 1 / slope,
        "latency_us": intercept * 1_000_000,
        "r_squared": 1 - residual / total,
    }


class HardwareCalibrationReferenceTests(unittest.TestCase):
    def test_calibrated_profile_and_replay_match_reference(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference_path = (
            root
            / "benchmarks"
            / "reference"
            / "hardware-rtx3080ti-p310"
            / "calibration.json"
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        self.assertEqual(
            reference["repository_base_commit_at_measurement"],
            "15dfc313f8995e7fdaee8023d52dad18c59f2f92",
        )
        self.assertEqual(
            reference["benchmark_implementation_commit"],
            "ee6a4ec8817449779d8dfb08bb560d0cd1be2661",
        )
        config = load_config(root / reference["calibrated_replay"]["config"])

        pinned_sync = reference["ram_to_vram"]["cases"]["pinned_sync"]
        p310_fit = reference["storage"]["devices"]["p310"]["linear_fit"]
        self.assertAlmostEqual(
            config.hardware.ram_to_vram_gbps,
            pinned_sync["linear_fit_bandwidth_gbps"],
            delta=0.01,
        )
        self.assertAlmostEqual(
            config.hardware.ram_latency_us,
            pinned_sync["linear_fit_latency_us"],
            delta=0.1,
        )
        self.assertAlmostEqual(
            config.hardware.nvme_to_ram_gbps,
            p310_fit["bandwidth_gbps"],
            delta=0.01,
        )
        self.assertAlmostEqual(
            config.hardware.nvme_latency_us,
            p310_fit["latency_us"],
            delta=0.1,
        )

        trace_paths = sorted(
            (
                root / "benchmarks" / "reference" / "real-routing-olmoe-m1" / "traces"
            ).glob("seed-*/*.trace.jsonl")
        )
        results = [
            compare_experiment(config, trace=read_trace(path)) for path in trace_paths
        ]
        baseline_ms = sum(result.baseline.elapsed_ms for result in results)
        prefetch_ms = sum(result.prefetch.elapsed_ms for result in results)
        replay = reference["calibrated_replay"]
        self.assertEqual(
            config.hardware.fixed_latency_scope,
            replay["fixed_latency_scope"],
        )
        self.assertEqual(len(results), replay["traces"])
        self.assertAlmostEqual(baseline_ms, replay["baseline_elapsed_ms"], places=9)
        self.assertAlmostEqual(prefetch_ms, replay["prefetch_elapsed_ms"], places=9)
        self.assertAlmostEqual(
            baseline_ms / prefetch_ms,
            replay["aggregate_speedup"],
            places=12,
        )

    def test_measurement_evidence_reconstructs_medians_and_fits(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference_dir = root / "benchmarks" / "reference" / "hardware-rtx3080ti-p310"
        calibration = json.loads(
            (reference_dir / "calibration.json").read_text(encoding="utf-8")
        )
        evidence = json.loads(
            (reference_dir / "measurement-evidence.json").read_text(encoding="utf-8")
        )

        # The checked-in evidence is intentionally safe to publish: raw host and
        # absolute target paths stay in the ignored local files.
        serialized = json.dumps(evidence)
        self.assertNotIn('"hostname"', serialized)
        self.assertNotIn("Itsme", serialized)
        self.assertNotIn("C:\\\\", serialized)
        self.assertNotIn("D:\\\\", serialized)

        raw_hashes = evidence["raw_files_sha256"]
        self.assertTrue(raw_hashes)
        for filename, digest in raw_hashes.items():
            self.assertEqual(Path(filename).name, filename)
            self.assertEqual(len(digest), 64)
            self.assertEqual(set(digest) - set("0123456789abcdef"), set())

        storage = evidence["storage"]
        storage_columns = storage["run_columns"]
        storage_series = {
            (entry["device"], entry["block_mib"], entry["queue_depth"]): [
                _row(storage_columns, run) for run in entry["runs"]
            ]
            for entry in storage["series"]
        }
        storage_metric_names = {
            "throughput_gbps": "median_throughput_gbps",
            "average_latency_ms": "median_average_latency_ms",
            "p50_latency_ms": "median_p50_latency_ms",
            "p95_latency_ms": "median_p95_latency_ms",
            "p99_latency_ms": "median_p99_latency_ms",
        }
        for device in ("p2", "p310"):
            expected_device = calibration["storage"]["devices"][device]
            for queue_depth in (1, 8):
                runs = storage_series[(device, 12, queue_depth)]
                expected = expected_device[f"queue_depth_{queue_depth}"]
                for source_name, expected_name in storage_metric_names.items():
                    actual = statistics.median(float(run[source_name]) for run in runs)
                    self.assertAlmostEqual(actual, expected[expected_name], places=12)

            points = []
            for block_mib in (1, 4, 12):
                runs = storage_series[(device, block_mib, 1)]
                points.append(
                    (
                        block_mib,
                        statistics.median(
                            float(run["average_latency_ms"]) for run in runs
                        ),
                    )
                )
            actual_fit = _linear_fit(points)
            expected_fit = expected_device["linear_fit"]
            for field in ("bandwidth_gbps", "latency_us", "r_squared"):
                self.assertAlmostEqual(
                    actual_fit[field],
                    expected_fit[field],
                    delta=abs(expected_fit[field]) * 1e-12 + 1e-12,
                )

        cuda = evidence["ram_to_vram"]
        cuda_columns = cuda["run_columns"]
        cuda_series = {
            (entry["case"], entry["block_mib"]): [
                _row(cuda_columns, run) for run in entry["runs"]
            ]
            for entry in cuda["series"]
        }
        cuda_metric_names = {
            "wall_throughput_gbps": "median_wall_throughput_gbps",
            "cuda_event_mean_ms": "median_cuda_event_mean_ms",
            "cuda_event_p50_ms": "median_cuda_event_p50_ms",
            "cuda_event_p95_ms": "median_cuda_event_p95_ms",
            "cuda_event_p99_ms": "median_cuda_event_p99_ms",
        }
        case_names = {
            "pageable-sync": "pageable_sync",
            "pinned-sync": "pinned_sync",
            "pinned-async": "pinned_async",
        }
        for evidence_name, calibration_name in case_names.items():
            expected = calibration["ram_to_vram"]["cases"][calibration_name]
            long_runs = cuda_series[(evidence_name, 12)]
            for source_name, expected_name in cuda_metric_names.items():
                actual = statistics.median(float(run[source_name]) for run in long_runs)
                self.assertAlmostEqual(actual, expected[expected_name], places=12)

            points = []
            for block_mib in (1, 4, 12):
                runs = cuda_series[(evidence_name, block_mib)]
                points.append(
                    (
                        block_mib,
                        statistics.median(
                            float(run["cuda_event_mean_ms"]) for run in runs
                        ),
                    )
                )
            actual_fit = _linear_fit(points)
            expected_fit = {
                "bandwidth_gbps": expected["linear_fit_bandwidth_gbps"],
                "latency_us": expected["linear_fit_latency_us"],
                "r_squared": expected["linear_fit_r_squared"],
            }
            for field in ("bandwidth_gbps", "latency_us", "r_squared"):
                self.assertAlmostEqual(
                    actual_fit[field],
                    expected_fit[field],
                    delta=abs(expected_fit[field]) * 1e-12 + 1e-12,
                )

        referenced_raw_files = {
            str(run["raw_file"])
            for runs in (*storage_series.values(), *cuda_series.values())
            for run in runs
        }
        self.assertEqual(referenced_raw_files, set(raw_hashes))

        # Locally, also verify that the compact record is anchored to the exact
        # ignored raw files. CI/source distributions need only the evidence JSON.
        raw_dir = root / "results" / "hardware-calibration" / "2026-08-10-p310"
        if raw_dir.is_dir():
            for filename, expected_digest in raw_hashes.items():
                actual_digest = hashlib.sha256(
                    (raw_dir / filename).read_bytes()
                ).hexdigest()
                self.assertEqual(actual_digest, expected_digest)


if __name__ == "__main__":
    unittest.main()
