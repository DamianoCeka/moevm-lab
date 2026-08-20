from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_default_model_remains_pinned_olmoe(self) -> None:
        self.assertEqual(_CAPTURE.DEFAULT_MODEL, "allenai/OLMoE-1B-7B-0924")
        self.assertEqual(
            _CAPTURE.DEFAULT_REVISION,
            "bd1c52f59153f724c1ad11ca1791edc77bab3806",
        )

    def test_qwen_request_requires_the_manifest_revision(self) -> None:
        manifest = _CAPTURE.qwen2_moe_assets.MANIFEST

        self.assertTrue(
            _CAPTURE._is_pinned_qwen_request(manifest.model_id, manifest.revision)
        )
        with self.assertRaisesRegex(ValueError, "requires the pinned revision"):
            _CAPTURE._is_pinned_qwen_request(manifest.model_id, "0" * 40)

    def test_qwen_uses_its_separate_checkpoint_license(self) -> None:
        manifest = _CAPTURE.qwen2_moe_assets.MANIFEST

        license_name, notice = _CAPTURE._checkpoint_license(
            manifest.model_id, manifest.revision
        )

        self.assertEqual(license_name, "Tongyi Qianwen License Agreement")
        self.assertIn("not covered by MoEVM Lab's Apache-2.0", notice)
        self.assertEqual(
            _CAPTURE._checkpoint_license(
                _CAPTURE.DEFAULT_MODEL, _CAPTURE.DEFAULT_REVISION
            ),
            ("Apache-2.0", None),
        )

    def test_custom_checkpoint_license_is_not_asserted(self) -> None:
        license_name, notice = _CAPTURE._checkpoint_license(
            "example/custom-moe", "f" * 40
        )

        self.assertEqual(license_name, "Unknown (not asserted)")
        self.assertIn("license was not verified", notice)
        self.assertNotIn("Apache-2.0", license_name)

    def test_custom_olmoe_revision_does_not_inherit_pinned_license(self) -> None:
        license_name, notice = _CAPTURE._checkpoint_license(
            _CAPTURE.DEFAULT_MODEL, "f" * 40
        )

        self.assertEqual(license_name, "Unknown (not asserted)")
        self.assertIsNotNone(notice)

    def test_qwen_verifies_the_complete_manifest_and_returns_shard_evidence(
        self,
    ) -> None:
        manifest = _CAPTURE.qwen2_moe_assets.MANIFEST
        verified = {
            filename: {
                "sha256": manifest.file_sha256[filename],
                "size_bytes": manifest.file_sizes[filename],
            }
            for filename in manifest.required_files
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / manifest.revision
            with (
                patch.object(
                    _CAPTURE.qwen2_moe_assets,
                    "verify_pinned_snapshot",
                    return_value=verified,
                ) as verifier,
                patch("builtins.print"),
            ):
                files, shards = _CAPTURE._verify_requested_checkpoint(
                    model_id=manifest.model_id,
                    revision=manifest.revision,
                    snapshot_path=snapshot,
                )

        verifier.assert_called_once_with(snapshot)
        self.assertEqual(files, dict(manifest.file_sha256))
        self.assertEqual(shards, _CAPTURE.qwen2_moe_assets.PINNED_SHARD_SHA256)
        self.assertIn("LICENSE", files)
        self.assertEqual(set(files), set(manifest.required_files))

    def test_qwen_rejects_incomplete_verification_evidence(self) -> None:
        manifest = _CAPTURE.qwen2_moe_assets.MANIFEST
        verified = {
            filename: {
                "sha256": manifest.file_sha256[filename],
                "size_bytes": manifest.file_sizes[filename],
            }
            for filename in manifest.required_files
            if filename != "LICENSE"
        }

        with (
            patch.object(
                _CAPTURE.qwen2_moe_assets,
                "verify_pinned_snapshot",
                return_value=verified,
            ),
            self.assertRaisesRegex(RuntimeError, "full manifest"),
        ):
            _CAPTURE._verify_requested_checkpoint(
                model_id=manifest.model_id,
                revision=manifest.revision,
                snapshot_path=Path(manifest.revision),
            )

    def test_olmoe_verification_path_is_unchanged(self) -> None:
        expected = {"model-00001-of-00003.safetensors": "a" * 64}
        snapshot = Path(_CAPTURE.DEFAULT_REVISION)

        with patch.object(
            _CAPTURE, "_verify_pinned_shards", return_value=expected
        ) as verifier:
            files, shards = _CAPTURE._verify_requested_checkpoint(
                model_id=_CAPTURE.DEFAULT_MODEL,
                revision=_CAPTURE.DEFAULT_REVISION,
                snapshot_path=snapshot,
            )

        verifier.assert_called_once_with(snapshot)
        self.assertIsNone(files)
        self.assertEqual(shards, expected)

    def test_qwen_normalizes_missing_resolved_revision_to_pin(self) -> None:
        manifest = _CAPTURE.qwen2_moe_assets.MANIFEST

        resolved = _CAPTURE._normalize_resolved_revision(
            SimpleNamespace(_commit_hash=None),
            model_id=manifest.model_id,
            requested_revision=manifest.revision,
        )

        self.assertEqual(resolved, manifest.revision)

    def test_qwen_rejects_resolved_revision_mismatch(self) -> None:
        manifest = _CAPTURE.qwen2_moe_assets.MANIFEST

        with self.assertRaisesRegex(RuntimeError, "resolved model revision"):
            _CAPTURE._normalize_resolved_revision(
                SimpleNamespace(_commit_hash="f" * 40),
                model_id=manifest.model_id,
                requested_revision=manifest.revision,
            )

    def test_olmoe_keeps_missing_resolved_revision_behavior(self) -> None:
        self.assertIsNone(
            _CAPTURE._normalize_resolved_revision(
                SimpleNamespace(_commit_hash=None),
                model_id=_CAPTURE.DEFAULT_MODEL,
                requested_revision=_CAPTURE.DEFAULT_REVISION,
            )
        )


if __name__ == "__main__":
    unittest.main()
