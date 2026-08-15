from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "render_cuda_overlap_timeline.py"
)
_SPEC = importlib.util.spec_from_file_location("render_cuda_overlap_timeline", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import CUDA timeline renderer: {_SCRIPT}")
_RENDERER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RENDERER
_SPEC.loader.exec_module(_RENDERER)


def _timeline(
    spans: list[dict[str, object]], *, status: str = "measured"
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "complete": True,
        "status": status,
        "method": "cuda_events_v1",
        "scope": "paged_expert_h2d_vs_expert_compute",
        "unit": "milliseconds",
        "spans": spans,
    }


def _payload() -> dict[str, object]:
    prefill = _timeline(
        [
            {
                "lane": "h2d",
                "name": "h2d:0:L0:E1",
                "sequence": 0,
                "layer": 0,
                "expert": 1,
                "start_ms": 1.0,
                "end_ms": 5.0,
            },
            {
                "lane": "expert_compute",
                "name": "expert_compute:1:L0:E2",
                "sequence": 1,
                "layer": 0,
                "expert": 2,
                "start_ms": 3.0,
                "end_ms": 8.0,
            },
            {
                "lane": "h2d",
                "name": "h2d:2:L1:E3",
                "sequence": 2,
                "layer": 1,
                "expert": 3,
                "start_ms": 6.0,
                "end_ms": 10.0,
            },
        ]
    )
    decode = _timeline(
        [
            {
                "lane": "h2d",
                "name": "h2d:0:L2:E4",
                "sequence": 0,
                "layer": 2,
                "expert": 4,
                "start_ms": 0.0,
                "end_ms": 2.0,
            },
            {
                "lane": "expert_compute",
                "name": "expert_compute:1:L2:E4",
                "sequence": 1,
                "layer": 2,
                "expert": 4,
                "start_ms": 2.0,
                "end_ms": 4.0,
            },
        ]
    )
    return {
        "schema_version": 1,
        "passes": {
            "cold_expert_cache": {
                "prefill": {"cuda_event_timeline": prefill},
                "decode": {
                    "per_token": [
                        {"index": 1, "cuda_event_timeline": decode},
                    ]
                },
            },
            "repeat_retained_expert_cache": {
                "prefill": {"cuda_event_timeline": prefill},
                "decode": {"per_token": []},
            },
        },
    }


class RenderCudaOverlapTimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_parser_accepts_required_call_forms(self) -> None:
        parser = _RENDERER.build_parser()
        prefill = parser.parse_args(
            [
                "--input",
                "result.json",
                "--pass",
                "cold_expert_cache",
                "--call",
                "prefill",
                "--output",
                "timeline.svg",
            ]
        )
        decode = parser.parse_args(
            [
                "--input",
                "result.json",
                "--pass",
                "repeat_retained_expert_cache",
                "--call",
                "decode:12",
                "--output",
                "timeline.svg",
            ]
        )

        self.assertEqual(prefill.selection.cli_value, "prefill")
        self.assertEqual(decode.selection.cli_value, "decode:12")
        with self.assertRaises(argparse.ArgumentTypeError):
            _RENDERER.parse_call("decode:-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            _RENDERER.parse_call("decode:one")

    def test_selects_prefill_and_renders_accessible_overlap_svg(self) -> None:
        selected = _RENDERER.select_timeline(
            _payload(),
            pass_name="cold_expert_cache",
            selection=_RENDERER.parse_call("prefill"),
        )

        rendered = _RENDERER.render_svg(selected)

        self.assertEqual(selected.status, "measured")
        self.assertIn('role="img"', rendered)
        self.assertIn('aria-labelledby="timeline-title timeline-desc"', rendered)
        self.assertIn('<title id="timeline-title">', rendered)
        self.assertIn('<desc id="timeline-desc">', rendered)
        self.assertIn("H2D transfer", rendered)
        self.assertIn("Expert compute", rendered)
        self.assertIn("L0 · E1", rendered)
        self.assertIn("L0 · E2", rendered)
        self.assertIn('class="overlap"', rendered)
        self.assertIn("Overlap highlighted: 4.000 ms", rendered)
        self.assertIn("do not establish physical NVMe activity", rendered)
        self.assertNotIn("<script", rendered)

    def test_decode_selection_uses_recorded_per_token_index(self) -> None:
        selected = _RENDERER.select_timeline(
            _payload(),
            pass_name="cold_expert_cache",
            selection=_RENDERER.parse_call("decode:1"),
        )

        rendered = _RENDERER.render_svg(selected)

        self.assertEqual(selected.selection.display_name, "decode token 1")
        self.assertIn("decode token 1", rendered)
        self.assertIn("Overlap highlighted: 0.000000 ms", rendered)
        with self.assertRaisesRegex(ValueError, "available decode indexes: 1"):
            _RENDERER.select_timeline(
                _payload(),
                pass_name="cold_expert_cache",
                selection=_RENDERER.parse_call("decode:0"),
            )

    def test_rejects_missing_or_incompatible_opt_in_telemetry(self) -> None:
        payload = _payload()
        del payload["passes"]["cold_expert_cache"]["prefill"]["cuda_event_timeline"]
        with self.assertRaisesRegex(ValueError, "cuda-overlap-telemetry"):
            _RENDERER.select_timeline(
                payload,
                pass_name="cold_expert_cache",
                selection=_RENDERER.parse_call("prefill"),
            )

        payload = _payload()
        payload["passes"]["cold_expert_cache"]["prefill"]["cuda_event_timeline"][
            "unit"
        ] = "seconds"
        with self.assertRaisesRegex(ValueError, "unit must be milliseconds"):
            _RENDERER.select_timeline(
                payload,
                pass_name="cold_expert_cache",
                selection=_RENDERER.parse_call("prefill"),
            )

        payload = _payload()
        payload["passes"]["cold_expert_cache"]["prefill"]["cuda_event_timeline"][
            "complete"
        ] = False
        with self.assertRaisesRegex(ValueError, "must be complete"):
            _RENDERER.select_timeline(
                payload,
                pass_name="cold_expert_cache",
                selection=_RENDERER.parse_call("prefill"),
            )

    def test_main_writes_create_only_svg(self) -> None:
        input_path = self.root / "result.json"
        output_path = self.root / "nested" / "timeline.svg"
        input_path.write_text(json.dumps(_payload()), encoding="utf-8")

        result = _RENDERER.main(
            [
                "--input",
                str(input_path),
                "--pass",
                "cold_expert_cache",
                "--call",
                "prefill",
                "--output",
                str(output_path),
            ]
        )

        self.assertEqual(result, 0)
        self.assertTrue(output_path.is_file())
        self.assertIn("CUDA overlap timeline", output_path.read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            _RENDERER.write_svg_create_only(output_path, "replacement")


if __name__ == "__main__":
    unittest.main()
