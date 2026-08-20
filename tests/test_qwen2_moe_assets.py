from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from moevm import qwen2_moe_assets


class Qwen2MoeAssetTests(unittest.TestCase):
    def test_manifest_pins_official_checkpoint_and_every_required_file(self) -> None:
        manifest = qwen2_moe_assets.MANIFEST

        self.assertEqual(manifest.model_id, "Qwen/Qwen1.5-MoE-A2.7B")
        self.assertEqual(
            manifest.revision,
            "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
        )
        self.assertEqual(
            manifest.required_files, qwen2_moe_assets.REQUIRED_SNAPSHOT_FILES
        )
        self.assertEqual(len(manifest.required_files), 18)
        self.assertEqual(len(qwen2_moe_assets.PINNED_SHARD_SHA256), 8)
        self.assertEqual(
            set(qwen2_moe_assets.PINNED_SHARD_SHA256),
            set(qwen2_moe_assets.PINNED_SHARD_SIZES),
        )
        self.assertEqual(
            qwen2_moe_assets.PINNED_SHARD_BYTES,
            28_632_144_944,
        )
        self.assertEqual(manifest.total_bytes, 28_644_048_831)

    def test_manifest_includes_license_model_index_and_all_shards(self) -> None:
        required = set(qwen2_moe_assets.MANIFEST.required_files)

        self.assertIn("LICENSE", required)
        self.assertIn("README.md", required)
        self.assertIn("config.json", required)
        self.assertIn("model.safetensors.index.json", required)
        self.assertEqual(
            {filename for filename in required if filename.endswith(".safetensors")},
            set(qwen2_moe_assets.PINNED_SHARD_SHA256),
        )

    def test_pinned_snapshot_path_delegates_without_loading_the_hub(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "hf-home"
            expected = Path(temporary) / "expected"
            resolver = Mock(return_value=expected)

            with patch.object(
                qwen2_moe_assets,
                "checkpoint_snapshot_path",
                resolver,
            ):
                actual = qwen2_moe_assets.pinned_snapshot_path(home)

        self.assertEqual(actual, expected)
        resolver.assert_called_once_with(qwen2_moe_assets.MANIFEST, home)

    def test_verify_pinned_snapshot_delegates_to_generic_verifier(self) -> None:
        snapshot = Path("snapshot")
        record = {"config.json": {"sha256": "0" * 64, "size_bytes": 1}}
        verifier = Mock(return_value=record)

        with patch.object(
            qwen2_moe_assets,
            "verify_checkpoint_snapshot",
            verifier,
        ):
            actual = qwen2_moe_assets.verify_pinned_snapshot(snapshot)

        self.assertIs(actual, record)
        verifier.assert_called_once_with(qwen2_moe_assets.MANIFEST, snapshot)

    def test_ensure_pinned_snapshot_forwards_explicit_authorization(self) -> None:
        home = Path("hf-home")
        expected = Path("snapshot")
        ensure = Mock(return_value=expected)

        with patch.object(
            qwen2_moe_assets,
            "ensure_checkpoint_snapshot",
            ensure,
        ):
            actual = qwen2_moe_assets.ensure_pinned_snapshot(
                home,
                allow_download=False,
            )

        self.assertIs(actual, expected)
        ensure.assert_called_once_with(
            qwen2_moe_assets.MANIFEST,
            home,
            allow_download=False,
        )


if __name__ == "__main__":
    unittest.main()
