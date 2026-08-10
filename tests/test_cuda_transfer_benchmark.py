from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from moevm.cuda_transfer_benchmark import (
    DEFAULT_CHUNK_BYTES,
    MAX_CHUNK_BYTES,
    _case_specs,
    _latency_summary,
    _validate_output_path,
    _validate_parameters,
    _verification_offsets,
    benchmark_cuda_transfers,
    parse_size,
)


class CudaTransferBenchmarkUnitTests(unittest.TestCase):
    def test_parse_size_accepts_binary_and_decimal_units(self) -> None:
        self.assertEqual(parse_size("12MiB"), DEFAULT_CHUNK_BYTES)
        self.assertEqual(parse_size("1.5MB"), 1_500_000)
        self.assertEqual(parse_size("4096"), 4096)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_size("zero")

    def test_latency_summary_reports_required_percentiles(self) -> None:
        summary = _latency_summary([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertAlmostEqual(summary["p99"], 3.97)

    def test_case_matrix_omits_misleading_pageable_async(self) -> None:
        names = [spec.name for spec in _case_specs("both")]
        self.assertEqual(names, ["pageable-sync", "pinned-sync", "pinned-async"])
        self.assertNotIn("pageable-async", names)

    def test_verification_samples_span_the_payload(self) -> None:
        self.assertEqual(_verification_offsets(8), (0, 2, 4, 7))
        self.assertEqual(_verification_offsets(1), (0,))

    def test_validation_enforces_bounded_allocations(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety limit"):
            _validate_parameters(
                chunk_bytes=MAX_CHUNK_BYTES + 1,
                operations=1,
                warmup_operations=0,
                async_depth=1,
                device_index=0,
                mode="both",
            )
        with self.assertRaisesRegex(ValueError, "async_depth"):
            _validate_parameters(
                chunk_bytes=DEFAULT_CHUNK_BYTES,
                operations=1,
                warmup_operations=0,
                async_depth=0,
                device_index=0,
                mode="both",
            )

    def test_output_path_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.json"
            self.assertEqual(_validate_output_path(output), output.resolve())
            output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _validate_output_path(output)


class CudaTransferBenchmarkCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("PyTorch is not installed") from exc
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA-enabled PyTorch and a CUDA GPU are required")
        cls.torch = torch

    def test_tiny_cuda_report_contains_verified_cases_and_percentiles(self) -> None:
        report = benchmark_cuda_transfers(
            chunk_bytes=1 * 1024 * 1024,
            operations=2,
            warmup_operations=1,
            async_depth=2,
            mode="both",
            torch_module=self.torch,
        )

        self.assertEqual(
            report["benchmark_type"],
            "host_to_device_cuda_transfer_microbenchmark",
        )
        self.assertEqual(len(report["results"]), 3)
        self.assertEqual(report["safety"]["peak_benchmark_device_payload_bytes"], 2**20)
        pinned_by_case = {
            result["name"]: result["source_is_pinned"] for result in report["results"]
        }
        self.assertFalse(pinned_by_case["pageable-sync"])
        self.assertTrue(pinned_by_case["pinned-sync"])
        self.assertTrue(pinned_by_case["pinned-async"])
        for result in report["results"]:
            self.assertTrue(result["verification"]["passed"])
            self.assertEqual(
                result["verification"]["device_sample_offsets"],
                [0, 262144, 524288, 1048575],
            )
            self.assertEqual(
                result["verification"]["device_sample_values_uint8"],
                [165, 165, 165, 165],
            )
            self.assertGreater(result["throughput"]["bytes_per_second"], 0)
            event_latency = result["latency_ms"]["cuda_event"]
            self.assertEqual(event_latency["samples"], 2)
            for percentile in ("p50", "p95", "p99"):
                self.assertGreaterEqual(event_latency[percentile], 0)


if __name__ == "__main__":
    unittest.main()
