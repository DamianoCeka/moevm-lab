from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from moevm.pinned_checkpoint import (
    PinnedCheckpointManifest,
    _download_without_windows_symlinks,
    _HubBindings,
    checkpoint_snapshot_path,
    ensure_checkpoint_snapshot,
    verify_checkpoint_snapshot,
)


class _LocalCacheMiss(OSError):
    pass


class _FakeApi:
    def __init__(self, files: tuple[str, ...]) -> None:
        self.files = files
        self.calls: list[tuple[str, str]] = []

    def list_repo_files(self, model_id: str, *, revision: str) -> tuple[str, ...]:
        self.calls.append((model_id, revision))
        return self.files


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class PinnedCheckpointManifestTests(unittest.TestCase):
    revision = "a" * 40

    def test_manifest_copies_and_freezes_a_complete_contract(self) -> None:
        digests = {"config.json": _digest(b"config")}
        sizes = {"config.json": 6}

        manifest = PinnedCheckpointManifest(
            model_id="owner/model_name",
            revision=self.revision,
            file_sha256=digests,
            file_sizes=sizes,
        )
        digests["config.json"] = "0" * 64
        sizes["config.json"] = 99

        self.assertEqual(manifest.required_files, ("config.json",))
        self.assertEqual(manifest.file_sha256["config.json"], _digest(b"config"))
        self.assertEqual(manifest.file_sizes["config.json"], 6)
        self.assertEqual(manifest.total_bytes, 6)
        with self.assertRaises(TypeError):
            manifest.file_sizes["config.json"] = 7  # type: ignore[index]

    def test_manifest_rejects_noncanonical_or_unsafe_model_ids(self) -> None:
        invalid = (
            "owner",
            "owner/repo/extra",
            "/repo",
            "owner/",
            "owner\\escape/repo",
            "owner/C:repo",
            "owner/../repo",
            "owner/.repo",
            "owner/repo.git",
            "owner/repo--copy",
            f"owner/{'x' * 91}",
        )
        for model_id in invalid:
            with self.subTest(model_id=model_id), self.assertRaises(ValueError):
                PinnedCheckpointManifest(
                    model_id=model_id,
                    revision=self.revision,
                    file_sha256={"file": "0" * 64},
                    file_sizes={"file": 1},
                )

    def test_manifest_requires_an_exact_commit_revision(self) -> None:
        for revision in ("main", "A" * 40, "a" * 39, "a" * 41):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                PinnedCheckpointManifest(
                    model_id="owner/repo",
                    revision=revision,
                    file_sha256={"file": "0" * 64},
                    file_sizes={"file": 1},
                )

    def test_manifest_rejects_mismatched_empty_or_malformed_mappings(self) -> None:
        invalid_contracts = (
            ({}, {}),
            ({"file": "0" * 64}, {}),
            ({"file": "0" * 63}, {"file": 1}),
            ({"file": "G" * 64}, {"file": 1}),
            ({"file": "0" * 64}, {"file": 0}),
            ({"file": "0" * 64}, {"file": -1}),
            ({"file": "0" * 64}, {"file": True}),
        )
        for digests, sizes in invalid_contracts:
            with (
                self.subTest(digests=digests, sizes=sizes),
                self.assertRaises(ValueError),
            ):
                PinnedCheckpointManifest(
                    model_id="owner/repo",
                    revision=self.revision,
                    file_sha256=digests,
                    file_sizes=sizes,
                )
        with self.assertRaises(ValueError):
            PinnedCheckpointManifest(
                model_id="owner/repo",
                revision=self.revision,
                file_sha256=None,  # type: ignore[arg-type]
                file_sizes=None,  # type: ignore[arg-type]
            )

    def test_manifest_rejects_noncanonical_or_escaping_filenames(self) -> None:
        invalid = (
            "",
            "/absolute",
            "../escape",
            "nested/../escape",
            "nested\\escape",
            "C:escape",
            "nested//file",
            "./file",
            "file/",
            "bad\x00name",
        )
        for filename in invalid:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                PinnedCheckpointManifest(
                    model_id="owner/repo",
                    revision=self.revision,
                    file_sha256={filename: "0" * 64},
                    file_sizes={filename: 1},
                )


class PinnedCheckpointTests(unittest.TestCase):
    revision = "b" * 40
    payloads: ClassVar[dict[str, bytes]] = {
        "config.json": b'{"model_type":"test"}',
        "nested/model.safetensors": b"checkpoint-weights",
    }

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name).resolve()
        self.manifest = PinnedCheckpointManifest(
            model_id="owner/repo",
            revision=self.revision,
            file_sha256={
                filename: _digest(payload)
                for filename, payload in self.payloads.items()
            },
            file_sizes={
                filename: len(payload) for filename, payload in self.payloads.items()
            },
        )

    def _snapshot(self, root: Path | None = None) -> Path:
        snapshot = (root or self.root) / self.revision
        for filename, payload in self.payloads.items():
            target = snapshot.joinpath(*filename.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return snapshot

    def _canonical_snapshot(self) -> Path:
        snapshot = checkpoint_snapshot_path(self.manifest, self.root)
        for filename, payload in self.payloads.items():
            target = snapshot.joinpath(*filename.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return snapshot

    @staticmethod
    def _bindings(
        download_snapshot,
        *,
        download_file=None,
        api_factory=None,
    ) -> _HubBindings:
        def unexpected(*_args, **_kwargs):
            raise AssertionError("unexpected Hub operation")

        return _HubBindings(
            api_factory=api_factory or unexpected,
            download_file=download_file or unexpected,
            download_snapshot=download_snapshot,
            local_entry_not_found_error=_LocalCacheMiss,
        )

    def test_canonical_snapshot_path_is_confined_below_hf_home(self) -> None:
        expected = (
            self.root / "hub" / "models--owner--repo" / "snapshots" / self.revision
        )
        actual = checkpoint_snapshot_path(self.manifest, self.root / ".")

        self.assertEqual(actual, expected)
        actual.relative_to(self.root)
        with self.assertRaises(ValueError):
            checkpoint_snapshot_path(self.manifest, "   ")

    def test_verify_checks_every_exact_size_and_digest(self) -> None:
        snapshot = self._snapshot()

        result = verify_checkpoint_snapshot(self.manifest, snapshot)

        self.assertEqual(tuple(result), self.manifest.required_files)
        for filename, payload in self.payloads.items():
            self.assertEqual(
                result[filename],
                {"sha256": _digest(payload), "size_bytes": len(payload)},
            )

    def test_verify_rejects_missing_file_wrong_revision_size_and_digest(self) -> None:
        snapshot = self._snapshot()
        missing = snapshot / "config.json"
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            verify_checkpoint_snapshot(self.manifest, snapshot)

        missing.write_bytes(self.payloads["config.json"] + b"x")
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            verify_checkpoint_snapshot(self.manifest, snapshot)

        missing.write_bytes(b"x" * len(self.payloads["config.json"]))
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            verify_checkpoint_snapshot(self.manifest, snapshot)

        wrong_revision = self.root / "not-the-pinned-revision"
        wrong_revision.mkdir()
        with self.assertRaisesRegex(ValueError, "pinned revision"):
            verify_checkpoint_snapshot(self.manifest, wrong_revision)

    def test_ensure_is_offline_first_and_does_not_use_network_on_cache_hit(
        self,
    ) -> None:
        expected = self._canonical_snapshot()
        calls: list[dict[str, object]] = []

        def download_snapshot(**kwargs):
            calls.append(kwargs)
            if kwargs["local_files_only"] is not True:
                raise AssertionError("network path used despite cache hit")
            return os.fspath(expected)

        bindings = self._bindings(download_snapshot)
        with patch(
            "moevm.pinned_checkpoint._load_huggingface_hub",
            return_value=bindings,
        ):
            actual = ensure_checkpoint_snapshot(
                self.manifest,
                self.root,
                allow_download=True,
            )

        self.assertEqual(actual, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["local_files_only"], True)
        self.assertEqual(calls[0]["repo_id"], self.manifest.model_id)
        self.assertEqual(calls[0]["revision"], self.manifest.revision)
        self.assertEqual(calls[0]["cache_dir"], self.root / "hub")
        self.assertEqual(calls[0]["allow_patterns"], self.manifest.required_files)

    def test_cache_miss_without_consent_never_attempts_download(self) -> None:
        calls: list[bool] = []

        def download_snapshot(**kwargs):
            calls.append(kwargs["local_files_only"])
            if kwargs["local_files_only"]:
                raise _LocalCacheMiss("not cached")
            raise AssertionError("download attempted without consent")

        bindings = self._bindings(download_snapshot)
        with (
            patch(
                "moevm.pinned_checkpoint._load_huggingface_hub",
                return_value=bindings,
            ),
            self.assertRaises(FileNotFoundError),
        ):
            ensure_checkpoint_snapshot(
                self.manifest,
                self.root,
                allow_download=False,
            )

        self.assertEqual(calls, [True])

    def test_download_requires_boolean_consent_before_loading_hub(self) -> None:
        with (
            patch("moevm.pinned_checkpoint._load_huggingface_hub") as load_hub,
            self.assertRaises(TypeError),
        ):
            ensure_checkpoint_snapshot(
                self.manifest,
                self.root,
                allow_download=1,  # type: ignore[arg-type]
            )
        load_hub.assert_not_called()

    def test_cache_miss_with_consent_downloads_pinned_required_files_only(
        self,
    ) -> None:
        expected = checkpoint_snapshot_path(self.manifest, self.root)
        calls: list[dict[str, object]] = []

        def download_snapshot(**kwargs):
            calls.append(kwargs)
            if kwargs["local_files_only"]:
                raise _LocalCacheMiss("not cached")
            for filename, payload in self.payloads.items():
                target = expected.joinpath(*filename.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            return expected

        bindings = self._bindings(download_snapshot)
        with patch(
            "moevm.pinned_checkpoint._load_huggingface_hub",
            return_value=bindings,
        ):
            actual = ensure_checkpoint_snapshot(
                self.manifest,
                self.root,
                allow_download=True,
            )

        self.assertEqual(actual, expected)
        self.assertEqual(
            [call["local_files_only"] for call in calls],
            [True, False],
        )
        self.assertTrue(
            all(
                call["allow_patterns"] == self.manifest.required_files for call in calls
            )
        )

    def test_unexpected_cached_or_downloaded_snapshot_path_is_rejected(self) -> None:
        outside = self.root / "unexpected"
        for allow_download in (False, True):
            calls = 0

            def download_snapshot(_allow_download=allow_download, **kwargs):
                nonlocal calls
                calls += 1
                if _allow_download and kwargs["local_files_only"]:
                    raise _LocalCacheMiss("not cached")
                return outside

            bindings = self._bindings(download_snapshot)
            with (
                self.subTest(allow_download=allow_download),
                patch(
                    "moevm.pinned_checkpoint._load_huggingface_hub",
                    return_value=bindings,
                ),
                self.assertRaisesRegex(RuntimeError, "unexpected snapshot path"),
            ):
                ensure_checkpoint_snapshot(
                    self.manifest,
                    self.root,
                    allow_download=allow_download,
                )
            self.assertEqual(calls, 2 if allow_download else 1)

    def test_non_windows_download_error_is_not_hidden_by_fallback(self) -> None:
        api_calls = 0

        def api_factory():
            nonlocal api_calls
            api_calls += 1
            return _FakeApi(())

        def download_snapshot(**kwargs):
            if kwargs["local_files_only"]:
                raise _LocalCacheMiss("not cached")
            raise OSError("ordinary failure")

        bindings = self._bindings(download_snapshot, api_factory=api_factory)
        with (
            patch(
                "moevm.pinned_checkpoint._load_huggingface_hub",
                return_value=bindings,
            ),
            patch(
                "moevm.pinned_checkpoint._is_windows_no_symlink_error",
                return_value=False,
            ),
            self.assertRaisesRegex(OSError, "ordinary failure"),
        ):
            ensure_checkpoint_snapshot(
                self.manifest,
                self.root,
                allow_download=True,
            )
        self.assertEqual(api_calls, 0)

    def test_windows_no_symlink_fallback_is_pinned_confined_and_verified(
        self,
    ) -> None:
        expected = checkpoint_snapshot_path(self.manifest, self.root)
        api = _FakeApi(self.manifest.required_files + ("README.md",))
        file_calls: list[dict[str, object]] = []

        def download_snapshot(**kwargs):
            if kwargs["local_files_only"]:
                raise _LocalCacheMiss("not cached")
            raise OSError("Windows symlink privilege")

        def download_file(**kwargs):
            file_calls.append(kwargs)
            filename = kwargs["filename"]
            target = Path(kwargs["local_dir"]).joinpath(*filename.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.payloads[filename])
            return os.fspath(target)

        bindings = self._bindings(
            download_snapshot,
            download_file=download_file,
            api_factory=lambda: api,
        )
        with (
            patch(
                "moevm.pinned_checkpoint._load_huggingface_hub",
                return_value=bindings,
            ),
            patch(
                "moevm.pinned_checkpoint._is_windows_no_symlink_error",
                return_value=True,
            ),
        ):
            actual = ensure_checkpoint_snapshot(
                self.manifest,
                self.root,
                allow_download=True,
            )

        self.assertEqual(actual, expected)
        self.assertEqual(api.calls, [(self.manifest.model_id, self.manifest.revision)])
        self.assertEqual(
            {call["filename"] for call in file_calls},
            set(self.manifest.required_files),
        )
        for call in file_calls:
            self.assertEqual(call["repo_id"], self.manifest.model_id)
            self.assertEqual(call["revision"], self.manifest.revision)
            self.assertEqual(call["local_dir"], expected)
            self.assertIs(call["local_files_only"], False)
            target = expected.joinpath(*call["filename"].split("/"))
            target.resolve().relative_to(expected.resolve())

    def test_windows_fallback_rejects_unsafe_upstream_and_missing_files(self) -> None:
        def unexpected_download_file(**_kwargs):
            raise AssertionError("unsafe or incomplete revision reached download")

        for files, message in (
            (self.manifest.required_files + ("../escape",), "unsafe"),
            (("config.json",), "missing required files"),
        ):
            api = _FakeApi(files)
            bindings = self._bindings(
                lambda **_kwargs: "unused",
                download_file=unexpected_download_file,
                api_factory=lambda api=api: api,
            )
            with (
                self.subTest(files=files),
                self.assertRaisesRegex((ValueError, RuntimeError), message),
            ):
                _download_without_windows_symlinks(
                    self.manifest,
                    bindings,
                    expected=checkpoint_snapshot_path(self.manifest, self.root),
                )

    def test_windows_fallback_rejects_unexpected_file_path(self) -> None:
        expected = checkpoint_snapshot_path(self.manifest, self.root)
        outside = self.root / "outside.bin"
        api = _FakeApi(self.manifest.required_files)

        def download_file(**kwargs):
            outside.write_bytes(self.payloads[kwargs["filename"]])
            return outside

        bindings = self._bindings(
            lambda **_kwargs: "unused",
            download_file=download_file,
            api_factory=lambda: api,
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected checkpoint file path"):
            _download_without_windows_symlinks(
                self.manifest,
                bindings,
                expected=expected,
            )

    def test_windows_fallback_rejects_a_resolved_parent_escape(self) -> None:
        filename = "nested/model.safetensors"
        payload = self.payloads[filename]
        manifest = PinnedCheckpointManifest(
            model_id="owner/repo",
            revision=self.revision,
            file_sha256={filename: _digest(payload)},
            file_sizes={filename: len(payload)},
        )
        expected = checkpoint_snapshot_path(manifest, self.root)
        outside = self.root / "outside"
        original_resolve = Path.resolve

        def resolve_with_escaped_parent(path: Path, *args, **kwargs) -> Path:
            if path == expected / "nested":
                return outside
            return original_resolve(path, *args, **kwargs)

        bindings = self._bindings(
            lambda **_kwargs: "unused",
            download_file=lambda **_kwargs: self.fail("escaped target downloaded"),
            api_factory=lambda: _FakeApi((filename,)),
        )
        with (
            patch(
                "moevm.pinned_checkpoint.Path.resolve",
                new=resolve_with_escaped_parent,
            ),
            self.assertRaisesRegex(RuntimeError, "escapes the snapshot"),
        ):
            _download_without_windows_symlinks(
                manifest,
                bindings,
                expected=expected,
            )


if __name__ == "__main__":
    unittest.main()
