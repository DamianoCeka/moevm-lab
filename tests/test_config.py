from __future__ import annotations

import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

from moevm import __version__
from moevm.cli import _bundled_config
from moevm.config import HardwareConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_version_metadata_is_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as handle:
            project_version = tomllib.load(handle)["project"]["version"]

        self.assertEqual(project_version, __version__)
        self.assertIn(
            f"version: {project_version}",
            (root / "CITATION.cff").read_text(encoding="utf-8"),
        )
        self.assertIn(
            f"## {project_version} —",
            (root / "CHANGELOG.md").read_text(encoding="utf-8"),
        )

    def test_loads_toy_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml")
        self.assertEqual(config.model.layers, 12)
        self.assertEqual(config.model.top_k, 4)
        self.assertGreater(config.hardware.vram_cache_bytes, 0)

    def test_rejects_non_finite_bandwidth(self) -> None:
        hardware = HardwareConfig(
            vram_cache_mib=1.0,
            ram_cache_mib=1.0,
            ram_to_vram_gbps=float("nan"),
            nvme_to_ram_gbps=1.0,
            ram_latency_us=0.0,
            nvme_latency_us=0.0,
            overlap_efficiency=1.0,
            prefetch_vram_fraction=0.0,
        )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            hardware.validate()

    def test_rejects_boolean_integer_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "toy.toml")
        invalid_model = replace(config.model, layers=True)

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            invalid_model.validate()

    def test_bundled_default_config_matches_repository_config(self) -> None:
        root = Path(__file__).resolve().parents[1]

        self.assertEqual(
            Path(_bundled_config("toy.toml")).read_bytes(),
            (root / "configs" / "toy.toml").read_bytes(),
        )
        self.assertEqual(
            Path(_bundled_config("k3_shape.toml")).read_bytes(),
            (root / "configs" / "k3_shape.toml").read_bytes(),
        )
        self.assertEqual(
            Path(_bundled_config("olmoe_1b_7b_0924.toml")).read_bytes(),
            (root / "configs" / "olmoe_1b_7b_0924.toml").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
