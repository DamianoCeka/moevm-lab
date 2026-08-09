from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from moevm.config import load_config
from moevm.trace import SyntheticRoutingTrace, read_trace, write_trace
from moevm.types import RoutingStep


class TraceTests(unittest.TestCase):
    def test_jsonl_round_trip(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml").with_tokens(2)
        original = list(SyntheticRoutingTrace(config).generate())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            write_trace(path, original)
            restored = read_trace(path)

        self.assertEqual(original, restored)

    def test_rejects_coercible_non_integer_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(
                '{"token":true,"layer":0,"experts":[0,1]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "token must be an integer"):
                read_trace(path)

    def test_rejects_non_array_experts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(
                '{"token":0,"layer":0,"experts":"01"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "experts must be a JSON array"):
                read_trace(path)

    def test_router_scores_round_trip(self) -> None:
        original = [RoutingStep(0, 0, (2, 5), (0.61, 0.27))]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scored.jsonl"
            write_trace(path, original)
            restored = read_trace(path)

        self.assertEqual(original, restored)

    def test_rejects_misaligned_router_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(
                '{"token":0,"layer":0,"experts":[0,1],"scores":[0.8]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scores must align"):
                read_trace(path)

    def test_rejects_invalid_router_probability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(
                '{"token":0,"layer":0,"experts":[0,1],"scores":[1.1,0.2]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "finite probability"):
                read_trace(path)


if __name__ == "__main__":
    unittest.main()
