from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from moevm import olmoe_assets


class _OfflineCacheMiss(FileNotFoundError):
    pass


class OlmoeAssetTests(unittest.TestCase):
    def _snapshot(self, root: Path) -> Path:
        return olmoe_assets.pinned_snapshot_path(root / "hf-home")

    def _tiny_contents(self) -> dict[str, bytes]:
        contents = {
            filename: f"metadata:{filename}".encode()
            for filename in olmoe_assets.REQUIRED_SNAPSHOT_FILES
        }
        for index, filename in enumerate(olmoe_assets.PINNED_SHARD_SHA256):
            contents[filename] = bytes([index + 1]) * (index + 2)
        return contents

    @contextmanager
    def _tiny_manifest(self, contents: dict[str, bytes]) -> Iterator[None]:
        sizes = {
            filename: len(contents[filename])
            for filename in olmoe_assets.PINNED_SHARD_SIZES
        }
        digests = {
            filename: hashlib.sha256(contents[filename]).hexdigest()
            for filename in olmoe_assets.PINNED_SHARD_SHA256
        }
        with (
            patch.object(olmoe_assets, "PINNED_SHARD_SIZES", sizes),
            patch.object(olmoe_assets, "PINNED_SHARD_SHA256", digests),
        ):
            yield

    @staticmethod
    def _write_snapshot(snapshot: Path, contents: dict[str, bytes]) -> None:
        snapshot.mkdir(parents=True)
        for filename, payload in contents.items():
            path = snapshot / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    @staticmethod
    def _bindings(
        *,
        download_snapshot: Mock,
        api_factory: Mock | None = None,
        download_file: Mock | None = None,
    ) -> olmoe_assets._HubBindings:
        return olmoe_assets._HubBindings(
            api_factory=api_factory or Mock(),
            download_file=download_file or Mock(),
            download_snapshot=download_snapshot,
            local_entry_not_found_error=_OfflineCacheMiss,
        )

    def test_pinned_snapshot_path_is_deterministic_and_import_free(self) -> None:
        loader = Mock(side_effect=AssertionError("Hub import must stay lazy"))
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(olmoe_assets, "_load_huggingface_hub", loader),
        ):
            home = Path(temporary) / "cache"
            expected = (
                home.resolve()
                / "hub"
                / "models--allenai--OLMoE-1B-7B-0924"
                / "snapshots"
                / olmoe_assets.PINNED_REVISION
            )

            self.assertEqual(olmoe_assets.pinned_snapshot_path(home), expected)
            loader.assert_not_called()

        with self.assertRaisesRegex(ValueError, "hf_home cannot be empty"):
            olmoe_assets.pinned_snapshot_path("")

    def test_verify_pinned_snapshot_checks_required_files_sizes_and_hashes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary))
            contents = self._tiny_contents()
            self._write_snapshot(snapshot, contents)

            with self._tiny_manifest(contents):
                verified = olmoe_assets.verify_pinned_snapshot(snapshot)

            self.assertEqual(set(verified), set(olmoe_assets.PINNED_SHARD_SHA256))
            for filename, record in verified.items():
                self.assertEqual(record["size_bytes"], len(contents[filename]))
                self.assertEqual(
                    record["sha256"], hashlib.sha256(contents[filename]).hexdigest()
                )

    def test_verify_rejects_missing_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary))
            contents = self._tiny_contents()
            self._write_snapshot(snapshot, contents)
            (snapshot / "tokenizer.json").unlink()

            with (
                self._tiny_manifest(contents),
                self.assertRaisesRegex(FileNotFoundError, "tokenizer.json"),
            ):
                olmoe_assets.verify_pinned_snapshot(snapshot)

    def test_verify_rejects_size_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary))
            contents = self._tiny_contents()
            self._write_snapshot(snapshot, contents)
            first_shard = next(iter(olmoe_assets.PINNED_SHARD_SIZES))
            wrong_sizes = {
                name: len(contents[name]) for name in olmoe_assets.PINNED_SHARD_SIZES
            }
            wrong_sizes[first_shard] += 1

            with (
                self._tiny_manifest(contents),
                patch.object(olmoe_assets, "PINNED_SHARD_SIZES", wrong_sizes),
                patch.object(olmoe_assets, "_sha256_stream") as sha256_stream,
                self.assertRaisesRegex(RuntimeError, "size mismatch"),
            ):
                olmoe_assets.verify_pinned_snapshot(snapshot)
            sha256_stream.assert_not_called()

    def test_verify_rejects_sha256_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary))
            contents = self._tiny_contents()
            self._write_snapshot(snapshot, contents)
            first_shard = next(iter(olmoe_assets.PINNED_SHARD_SHA256))

            with self._tiny_manifest(contents):
                wrong_hashes = dict(olmoe_assets.PINNED_SHARD_SHA256)
                wrong_hashes[first_shard] = "0" * 64
                with (
                    patch.object(olmoe_assets, "PINNED_SHARD_SHA256", wrong_hashes),
                    self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"),
                ):
                    olmoe_assets.verify_pinned_snapshot(snapshot)

    def test_ensure_uses_verified_offline_cache_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "hf-home"
            snapshot = olmoe_assets.pinned_snapshot_path(home)
            contents = self._tiny_contents()
            self._write_snapshot(snapshot, contents)
            download_snapshot = Mock(return_value=str(snapshot))
            bindings = self._bindings(download_snapshot=download_snapshot)

            with (
                self._tiny_manifest(contents),
                patch.object(
                    olmoe_assets, "_load_huggingface_hub", return_value=bindings
                ),
            ):
                resolved = olmoe_assets.ensure_pinned_snapshot(
                    home, allow_download=True
                )

            self.assertEqual(resolved, snapshot.resolve())
            self.assertEqual(download_snapshot.call_count, 1)
            self.assertTrue(download_snapshot.call_args.kwargs["local_files_only"])

    def test_ensure_refuses_download_when_offline_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "hf-home"
            download_snapshot = Mock(side_effect=_OfflineCacheMiss("missing"))
            bindings = self._bindings(download_snapshot=download_snapshot)

            with (
                patch.object(
                    olmoe_assets, "_load_huggingface_hub", return_value=bindings
                ),
                self.assertRaisesRegex(FileNotFoundError, "not available offline"),
            ):
                olmoe_assets.ensure_pinned_snapshot(home, allow_download=False)

            self.assertEqual(download_snapshot.call_count, 1)
            self.assertTrue(download_snapshot.call_args.kwargs["local_files_only"])
            self.assertFalse(home.exists())

    def test_ensure_downloads_pinned_revision_after_offline_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "hf-home"
            snapshot = olmoe_assets.pinned_snapshot_path(home)
            contents = self._tiny_contents()
            self._write_snapshot(snapshot, contents)
            download_snapshot = Mock(
                side_effect=(_OfflineCacheMiss("missing"), str(snapshot))
            )
            bindings = self._bindings(download_snapshot=download_snapshot)

            with (
                self._tiny_manifest(contents),
                patch.object(
                    olmoe_assets, "_load_huggingface_hub", return_value=bindings
                ),
            ):
                resolved = olmoe_assets.ensure_pinned_snapshot(
                    home, allow_download=True
                )

            self.assertEqual(resolved, snapshot.resolve())
            self.assertEqual(download_snapshot.call_count, 2)
            first, second = download_snapshot.call_args_list
            self.assertTrue(first.kwargs["local_files_only"])
            self.assertFalse(second.kwargs["local_files_only"])
            self.assertEqual(second.kwargs["repo_id"], olmoe_assets.PINNED_MODEL_ID)
            self.assertEqual(second.kwargs["revision"], olmoe_assets.PINNED_REVISION)

    def test_ensure_uses_windows_no_symlink_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "hf-home"
            snapshot = olmoe_assets.pinned_snapshot_path(home)
            contents = self._tiny_contents()
            no_symlink = OSError("privilege not held")
            no_symlink.winerror = 1314  # type: ignore[attr-defined]
            download_snapshot = Mock(
                side_effect=(_OfflineCacheMiss("missing"), no_symlink)
            )
            api = Mock()
            api.list_repo_files.return_value = [
                *olmoe_assets.REQUIRED_SNAPSHOT_FILES,
                "legacy.bin",
            ]
            api_factory = Mock(return_value=api)

            def write_downloaded_file(**kwargs: object) -> str:
                filename = str(kwargs["filename"])
                destination = Path(str(kwargs["local_dir"])) / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents[filename])
                return str(destination)

            download_file = Mock(side_effect=write_downloaded_file)
            bindings = self._bindings(
                download_snapshot=download_snapshot,
                api_factory=api_factory,
                download_file=download_file,
            )

            with (
                self._tiny_manifest(contents),
                patch.object(
                    olmoe_assets, "_load_huggingface_hub", return_value=bindings
                ),
                patch.object(
                    olmoe_assets,
                    "_is_windows_no_symlink_error",
                    return_value=True,
                ),
            ):
                resolved = olmoe_assets.ensure_pinned_snapshot(
                    home, allow_download=True
                )

            self.assertEqual(resolved, snapshot.resolve())
            downloaded_names = {
                call.kwargs["filename"] for call in download_file.call_args_list
            }
            self.assertEqual(
                downloaded_names, set(olmoe_assets.REQUIRED_SNAPSHOT_FILES)
            )
            api.list_repo_files.assert_called_once_with(
                olmoe_assets.PINNED_MODEL_ID,
                revision=olmoe_assets.PINNED_REVISION,
            )

    def test_ensure_rejects_unexpected_snapshot_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "hf-home"
            unexpected = Path(temporary) / olmoe_assets.PINNED_REVISION
            unexpected.mkdir()
            bindings = self._bindings(
                download_snapshot=Mock(return_value=str(unexpected))
            )

            with (
                patch.object(
                    olmoe_assets, "_load_huggingface_hub", return_value=bindings
                ),
                self.assertRaisesRegex(RuntimeError, "unexpected snapshot path"),
            ):
                olmoe_assets.ensure_pinned_snapshot(home, allow_download=False)

    def test_ensure_rejects_non_boolean_download_authorization(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            olmoe_assets.ensure_pinned_snapshot("cache", allow_download=1)  # type: ignore[arg-type]

    def test_windows_fallback_rejects_drive_paths_and_alternate_streams(self) -> None:
        unsafe_names = (
            "C:/outside/file.safetensors",
            "C:outside/file.safetensors",
            "model.safetensors:stream",
            "\\\\server\\share\\model.safetensors",
        )
        for filename in unsafe_names:
            with (
                self.subTest(filename=filename),
                self.assertRaisesRegex(RuntimeError, "unsafe repository filename"),
            ):
                olmoe_assets._safe_repo_filename(filename)


if __name__ == "__main__":
    unittest.main()
