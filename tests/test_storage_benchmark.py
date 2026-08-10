from __future__ import annotations

import argparse
import json
import random
import tempfile
import unittest
from pathlib import Path

from moevm.storage_benchmark import (
    _percentile,
    _random_aligned_offset,
    _validate_output_path,
    benchmark_storage,
    main,
    parse_size,
)


class StorageBenchmarkTests(unittest.TestCase):
    def test_parse_size_accepts_binary_and_decimal_suffixes(self) -> None:
        self.assertEqual(parse_size("12MiB"), 12 * 1024 * 1024)
        self.assertEqual(parse_size("1.5MB"), 1_500_000)
        self.assertEqual(parse_size("4096"), 4096)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_size("1.2 bytes")

    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(_percentile(values, 0.50), 2.5)
        self.assertAlmostEqual(_percentile(values, 0.95), 3.85)

    def test_random_offset_is_seeded_aligned_and_in_bounds(self) -> None:
        first = random.Random(42)
        second = random.Random(42)
        offsets_a = [
            _random_aligned_offset(
                first,
                file_size=64 * 1024,
                chunk_bytes=4096,
                alignment_bytes=4096,
            )
            for _ in range(10)
        ]
        offsets_b = [
            _random_aligned_offset(
                second,
                file_size=64 * 1024,
                chunk_bytes=4096,
                alignment_bytes=4096,
            )
            for _ in range(10)
        ]
        self.assertEqual(offsets_a, offsets_b)
        self.assertTrue(all(offset % 4096 == 0 for offset in offsets_a))
        self.assertTrue(all(offset <= 60 * 1024 for offset in offsets_a))

    def test_buffered_benchmark_reports_reads_and_preserves_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "tiny-shard.bin"
            original = bytes(range(256)) * 256
            target.write_bytes(original)
            before = target.stat()

            report = benchmark_storage(
                [target],
                chunk_bytes=4096,
                operations_per_target=3,
                warmup_operations_per_target=1,
                seed=7,
                io_mode="buffered",
                alignment_bytes=4096,
            )

            after = target.stat()
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(after.st_size, before.st_size)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(report["benchmark_type"], "microbenchmark")
            self.assertEqual(report["summary"]["operations"], 3)
            self.assertEqual(report["summary"]["bytes_read"], 3 * 4096)
            self.assertEqual(report["summary"]["warmup"]["operations"], 1)
            self.assertFalse(report["cache_policy"]["os_page_cache_bypassed"])
            self.assertFalse(report["cache_policy"]["cold_cache_guaranteed"])
            self.assertEqual(report["safety"]["target_access"], "read_only")
            self.assertEqual(report["safety"]["target_writes"], 0)
            self.assertTrue(report["summary"]["all_targets_unchanged"])
            self.assertGreaterEqual(report["summary"]["latency_ms"]["p99"], 0)

    def test_rejects_target_smaller_than_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "small.bin"
            target.write_bytes(b"x" * 1024)
            with self.assertRaisesRegex(ValueError, "smaller"):
                benchmark_storage(
                    [target],
                    chunk_bytes=4096,
                    operations_per_target=1,
                    warmup_operations_per_target=0,
                )

    def test_output_cannot_replace_target_or_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target.bin"
            target.write_bytes(b"data")
            with self.assertRaisesRegex(ValueError, "cannot be a benchmark target"):
                _validate_output_path(target, [str(target)])

            output = Path(temp_dir) / "existing.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _validate_output_path(output, [str(target)])

    def test_cli_creates_valid_json_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "tiny-shard.bin"
            target.write_bytes(bytes(range(256)) * 256)
            output = Path(temp_dir) / "report.json"

            exit_code = main(
                [
                    str(target),
                    "--chunk-size",
                    "4096",
                    "--operations",
                    "1",
                    "--warmup",
                    "0",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["operations"], 1)
            self.assertEqual(main([str(target), "--output", str(output)]), 2)


if __name__ == "__main__":
    unittest.main()
