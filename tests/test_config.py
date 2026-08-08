from __future__ import annotations

import unittest
from pathlib import Path

from moevm.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_toy_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml")
        self.assertEqual(config.model.layers, 12)
        self.assertEqual(config.model.top_k, 4)
        self.assertGreater(config.hardware.vram_cache_bytes, 0)


if __name__ == "__main__":
    unittest.main()
