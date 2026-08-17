"""Generic acquisition and integrity checks for pinned Hub checkpoints.

Weights remain third-party assets in the user's Hugging Face cache.  The helper
is intentionally model-agnostic, lazy-imports :mod:`huggingface_hub`, never
loads model code, and verifies every required file before returning it.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

_HASH_BLOCK_BYTES = 16 * 1024 * 1024
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?$")


def _safe_repo_filename(filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError("checkpoint manifest contains an invalid filename")
    posix_path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)
    if (
        posix_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or "\\" in filename
        or ":" in filename
        or "\x00" in filename
        or any(part in ("", ".", "..") for part in posix_path.parts)
        or filename != posix_path.as_posix()
    ):
        raise ValueError(f"unsafe checkpoint filename: {filename!r}")
    return filename


def _valid_model_id(model_id: object) -> bool:
    if not isinstance(model_id, str) or len(model_id) > 96:
        return False
    components = model_id.split("/")
    return (
        len(components) == 2
        and all(_REPO_COMPONENT.fullmatch(component) for component in components)
        and "--" not in model_id
        and ".." not in model_id
        and not model_id.endswith(".git")
    )


@dataclass(frozen=True, slots=True)
class PinnedCheckpointManifest:
    """Immutable identity, size, and digest contract for one Hub revision."""

    model_id: str
    revision: str
    file_sha256: Mapping[str, str]
    file_sizes: Mapping[str, int]

    def __post_init__(self) -> None:
        if not _valid_model_id(self.model_id):
            raise ValueError("model_id must be a canonical owner/repository name")
        if (
            not isinstance(self.revision, str)
            or _HEX_40.fullmatch(self.revision) is None
        ):
            raise ValueError("revision must be a lowercase 40-hex commit")
        try:
            digests = dict(self.file_sha256)
            sizes = dict(self.file_sizes)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint file manifests must be mappings") from exc
        if not digests or set(digests) != set(sizes):
            raise ValueError("checkpoint digest and size manifests must match")
        for filename, digest in digests.items():
            _safe_repo_filename(filename)
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                raise ValueError(f"invalid SHA-256 for checkpoint file {filename}")
            size = sizes[filename]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"invalid size for checkpoint file {filename}")
        object.__setattr__(self, "file_sha256", MappingProxyType(digests))
        object.__setattr__(self, "file_sizes", MappingProxyType(sizes))

    @property
    def required_files(self) -> tuple[str, ...]:
        return tuple(sorted(self.file_sha256))

    @property
    def total_bytes(self) -> int:
        return sum(self.file_sizes.values())


@dataclass(frozen=True, slots=True)
class _HubBindings:
    api_factory: Callable[[], Any]
    download_file: Callable[..., str]
    download_snapshot: Callable[..., str]
    local_entry_not_found_error: type[OSError]


def _load_huggingface_hub() -> _HubBindings:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "checkpoint acquisition requires the pinned real-traces dependencies; "
            "install `moevm-lab[real-traces]` first"
        ) from exc
    return _HubBindings(
        api_factory=HfApi,
        download_file=hf_hub_download,
        download_snapshot=snapshot_download,
        local_entry_not_found_error=LocalEntryNotFoundError,
    )


def _normalized_path(raw_path: str | os.PathLike[str], *, name: str) -> Path:
    if isinstance(raw_path, str) and not raw_path.strip():
        raise ValueError(f"{name} cannot be empty")
    try:
        return Path(raw_path).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a filesystem path") from exc


def checkpoint_snapshot_path(
    manifest: PinnedCheckpointManifest,
    hf_home: str | os.PathLike[str],
) -> Path:
    """Return the canonical Hub-cache path for one pinned manifest."""

    home = _normalized_path(hf_home, name="hf_home")
    repository_folder = f"models--{manifest.model_id.replace('/', '--')}"
    return home / "hub" / repository_folder / "snapshots" / manifest.revision


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_snapshot(
    manifest: PinnedCheckpointManifest,
    snapshot: str | os.PathLike[str],
) -> dict[str, dict[str, int | str]]:
    """Verify the exact size and SHA-256 of every required checkpoint file."""

    resolved = _normalized_path(snapshot, name="snapshot")
    if not resolved.is_dir():
        raise FileNotFoundError(f"pinned snapshot directory not found: {resolved}")
    if resolved.name != manifest.revision:
        raise ValueError(
            f"snapshot directory must be the pinned revision {manifest.revision}"
        )

    verified: dict[str, dict[str, int | str]] = {}
    for filename in manifest.required_files:
        path = resolved.joinpath(*PurePosixPath(filename).parts)
        if not path.is_file():
            raise FileNotFoundError(f"pinned snapshot file not found: {path}")
        actual_size = path.stat().st_size
        expected_size = manifest.file_sizes[filename]
        if actual_size != expected_size:
            raise RuntimeError(
                f"checkpoint size mismatch for {filename}: "
                f"{actual_size} != {expected_size}"
            )
        actual_digest = _sha256_stream(path)
        expected_digest = manifest.file_sha256[filename]
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch for {filename}: "
                f"{actual_digest} != {expected_digest}"
            )
        verified[filename] = {
            "sha256": actual_digest,
            "size_bytes": actual_size,
        }
    return verified


def _validated_download_path(
    manifest: PinnedCheckpointManifest,
    downloaded: object,
    *,
    expected: Path,
) -> Path:
    if not isinstance(downloaded, (str, os.PathLike)):
        raise RuntimeError("Hugging Face returned an invalid snapshot path")
    resolved = Path(downloaded).expanduser().resolve()
    if resolved != expected.resolve():
        raise RuntimeError(
            "Hugging Face returned an unexpected snapshot path: "
            f"{resolved} != {expected.resolve()}"
        )
    verify_checkpoint_snapshot(manifest, resolved)
    return resolved


def _is_windows_no_symlink_error(exc: OSError) -> bool:
    return os.name == "nt" and getattr(exc, "winerror", None) == 1314


def _fallback_target(expected: Path, filename: str) -> Path:
    """Resolve a no-symlink fallback target without permitting cache escape."""

    root = expected.resolve()
    target = expected.joinpath(*PurePosixPath(filename).parts)
    try:
        target.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"checkpoint fallback target escapes the snapshot: {filename}"
        ) from exc
    if target.is_symlink():
        raise RuntimeError(
            f"checkpoint fallback target must not be a symlink: {filename}"
        )
    return target


def _validate_downloaded_file(downloaded: object, *, target: Path) -> None:
    if not isinstance(downloaded, (str, os.PathLike)):
        raise RuntimeError("Hugging Face returned an invalid checkpoint file path")
    resolved = Path(downloaded).expanduser().resolve()
    expected = target.resolve()
    if resolved != expected:
        raise RuntimeError(
            "Hugging Face returned an unexpected checkpoint file path: "
            f"{resolved} != {expected}"
        )


def _download_without_windows_symlinks(
    manifest: PinnedCheckpointManifest,
    bindings: _HubBindings,
    *,
    expected: Path,
) -> Path:
    expected.mkdir(parents=True, exist_ok=True)
    filenames = tuple(
        _safe_repo_filename(filename)
        for filename in bindings.api_factory().list_repo_files(
            manifest.model_id,
            revision=manifest.revision,
        )
    )
    missing_upstream = set(manifest.required_files) - set(filenames)
    if missing_upstream:
        raise RuntimeError(
            "pinned Hub revision is missing required files: "
            + ", ".join(sorted(missing_upstream))
        )
    for filename in manifest.required_files:
        target = _fallback_target(expected, filename)
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = bindings.download_file(
            repo_id=manifest.model_id,
            filename=filename,
            revision=manifest.revision,
            local_dir=expected,
            local_files_only=False,
        )
        target = _fallback_target(expected, filename)
        _validate_downloaded_file(downloaded, target=target)
    return expected


def ensure_checkpoint_snapshot(
    manifest: PinnedCheckpointManifest,
    hf_home: str | os.PathLike[str],
    *,
    allow_download: bool,
) -> Path:
    """Return a verified snapshot, downloading only when explicitly allowed."""

    if not isinstance(allow_download, bool):
        raise TypeError("allow_download must be a boolean")
    expected = checkpoint_snapshot_path(manifest, hf_home)
    hub_cache = expected.parents[2]
    bindings = _load_huggingface_hub()
    arguments = {
        "repo_id": manifest.model_id,
        "revision": manifest.revision,
        "cache_dir": hub_cache,
        "allow_patterns": manifest.required_files,
    }

    try:
        cached = bindings.download_snapshot(
            **arguments,
            local_files_only=True,
        )
    except bindings.local_entry_not_found_error as offline_error:
        if not allow_download:
            raise FileNotFoundError(
                f"the verified pinned snapshot is unavailable at {expected}"
            ) from offline_error
    else:
        return _validated_download_path(manifest, cached, expected=expected)

    try:
        downloaded = bindings.download_snapshot(
            **arguments,
            local_files_only=False,
        )
    except OSError as exc:
        if not _is_windows_no_symlink_error(exc):
            raise
        downloaded = _download_without_windows_symlinks(
            manifest,
            bindings,
            expected=expected,
        )
    return _validated_download_path(manifest, downloaded, expected=expected)
