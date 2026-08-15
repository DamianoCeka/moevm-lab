from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "capture_real_routing.py"
_SPEC = importlib.util.spec_from_file_location("capture_real_routing", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError(f"cannot import capture harness: {_SCRIPT}")
_CAPTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CAPTURE)


class CaptureRealRoutingTests(unittest.TestCase):
    def test_serialized_device_map_preserves_accelerate_dispatch(self) -> None:
        model = SimpleNamespace(
            hf_device_map={"model.layers.0": 0, "lm_head": "cpu"},
            device="cuda:0",
        )

        self.assertEqual(
            _CAPTURE._serialized_device_map(model),
            {"model.layers.0": "0", "lm_head": "cpu"},
        )

    def test_serialized_device_map_supports_fully_resident_model(self) -> None:
        model = SimpleNamespace(device="cuda:0")

        self.assertEqual(_CAPTURE._serialized_device_map(model), {"": "cuda:0"})

    def test_serialized_device_map_can_be_absent(self) -> None:
        self.assertEqual(_CAPTURE._serialized_device_map(SimpleNamespace()), {})


if __name__ == "__main__":
    unittest.main()
