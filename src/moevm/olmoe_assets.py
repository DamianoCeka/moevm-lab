"""Pinned OLMoE checkpoint acquisition and integrity verification.

The model weights remain third-party assets in the user's Hugging Face cache.
This module deliberately imports :mod:`huggingface_hub` only when acquisition
is requested so package import and plan-only workflows stay lightweight.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PINNED_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PINNED_REVISION = "bd1c52f59153f724c1ad11ca1791edc77bab3806"
PINNED_SHARD_SHA256 = {
    "model-00001-of-00003.safetensors": (
        "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"
    ),
    "model-00002-of-00003.safetensors": (
        "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"
    ),
    "model-00003-of-00003.safetensors": (
        "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"
    ),
}
PINNED_SHARD_SIZES = {
    "model-00001-of-00003.safetensors": 4_997_744_872,
    "model-00002-of-00003.safetensors": 4_997_235_176,
    "model-00003-of-00003.safetensors": 3_843_741_912,
}
PINNED_SHARD_BYTES = sum(PINNED_SHARD_SIZES.values())

_REQUIRED_METADATA_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
REQUIRED_SNAPSHOT_FILES = _REQUIRED_METADATA_FILES + tuple(PINNED_SHARD_SHA256)
_IGNORED_CHECKPOINT_SUFFIXES = (".bin", ".h5", ".msgpack", ".ot")
_HASH_BLOCK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
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


def pinned_snapshot_path(hf_home: str | os.PathLike[str]) -> Path:
    """Return the canonical Hub-cache path for the pinned OLMoE revision."""

    home = _normalized_path(hf_home, name="hf_home")
    repository_folder = f"models--{PINNED_MODEL_ID.replace('/', '--')}"
    return home / "hub" / repository_folder / "snapshots" / PINNED_REVISION


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_snapshot(
    snapshot: str | os.PathLike[str],
) -> dict[str, dict[str, int | str]]:
    """Verify required files plus exact size and SHA-256 for every weight shard."""

    resolved = _normalized_path(snapshot, name="snapshot")
    if not resolved.is_dir():
        raise FileNotFoundError(f"pinned snapshot directory not found: {resolved}")
    if resolved.name != PINNED_REVISION:
        raise ValueError(
            f"snapshot directory must be the pinned revision {PINNED_REVISION}"
        )
    for filename in REQUIRED_SNAPSHOT_FILES:
        required_path = resolved / filename
        if not required_path.is_file():
            raise FileNotFoundError(f"pinned snapshot file not found: {required_path}")

    if set(PINNED_SHARD_SHA256) != set(PINNED_SHARD_SIZES):
        raise RuntimeError("pinned checkpoint hash and size manifests disagree")

    verified: dict[str, dict[str, int | str]] = {}
    for filename, expected_digest in PINNED_SHARD_SHA256.items():
        shard_path = resolved / filename
        actual_size = shard_path.stat().st_size
        expected_size = PINNED_SHARD_SIZES[filename]
        if actual_size != expected_size:
            raise RuntimeError(
                f"checkpoint size mismatch for {filename}: "
                f"{actual_size} != {expected_size}"
            )
        actual_digest = _sha256_stream(shard_path)
        if actual_digest.lower() != expected_digest.lower():
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch for {filename}: "
                f"{actual_digest} != {expected_digest}"
            )
        verified[filename] = {
            "sha256": actual_digest,
            "size_bytes": actual_size,
        }
    return verified


def _snapshot_download_arguments(
    hub_cache: Path, *, local_files_only: bool
) -> dict[str, object]:
    return {
        "repo_id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "cache_dir": hub_cache,
        "local_files_only": local_files_only,
        "ignore_patterns": tuple(
            f"*{suffix}" for suffix in _IGNORED_CHECKPOINT_SUFFIXES
        ),
    }


def _validated_download_path(downloaded: object, *, expected: Path) -> Path:
    if not isinstance(downloaded, (str, os.PathLike)):
        raise RuntimeError("Hugging Face returned an invalid snapshot path")
    resolved = Path(downloaded).expanduser().resolve()
    if resolved != expected.resolve():
        raise RuntimeError(
            "Hugging Face returned an unexpected snapshot path: "
            f"{resolved} != {expected.resolve()}"
        )
    verify_pinned_snapshot(resolved)
    return resolved


def _is_windows_no_symlink_error(exc: OSError) -> bool:
    return os.name == "nt" and getattr(exc, "winerror", None) == 1314


def _safe_repo_filename(filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        raise RuntimeError("Hugging Face returned an invalid repository filename")
    posix_path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)
    if (
        posix_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or "\\" in filename
        or ":" in filename
        or any(part in ("", ".", "..") for part in posix_path.parts)
    ):
        raise RuntimeError(f"unsafe repository filename: {filename!r}")
    return filename


def _download_without_windows_symlinks(
    bindings: _HubBindings, *, expected: Path
) -> Path:
    expected.mkdir(parents=True, exist_ok=True)
    filenames = tuple(
        _safe_repo_filename(filename)
        for filename in bindings.api_factory().list_repo_files(
            PINNED_MODEL_ID,
            revision=PINNED_REVISION,
        )
    )
    missing_upstream = set(REQUIRED_SNAPSHOT_FILES) - set(filenames)
    if missing_upstream:
        raise RuntimeError(
            "pinned Hub revision is missing required files: "
            + ", ".join(sorted(missing_upstream))
        )
    for filename in filenames:
        if filename.endswith(_IGNORED_CHECKPOINT_SUFFIXES):
            continue
        target = expected / filename
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        bindings.download_file(
            repo_id=PINNED_MODEL_ID,
            filename=filename,
            revision=PINNED_REVISION,
            local_dir=expected,
            local_files_only=False,
        )
    return expected


def ensure_pinned_snapshot(
    hf_home: str | os.PathLike[str], allow_download: bool
) -> Path:
    """Return a verified pinned snapshot, downloading only when authorized.

    The Hub cache is checked with ``local_files_only=True`` first. A missing or
    incomplete cache may be populated only when ``allow_download`` is true.
    Corrupt cached content fails verification instead of being silently replaced.
    """

    if not isinstance(allow_download, bool):
        raise TypeError("allow_download must be a boolean")
    expected = pinned_snapshot_path(hf_home)
    hub_cache = expected.parents[2]
    bindings = _load_huggingface_hub()

    try:
        cached = bindings.download_snapshot(
            **_snapshot_download_arguments(hub_cache, local_files_only=True)
        )
    except bindings.local_entry_not_found_error as offline_error:
        if not allow_download:
            raise FileNotFoundError(
                "the verified pinned OLMoE snapshot is not available offline at "
                f"{expected}"
            ) from offline_error
    else:
        return _validated_download_path(cached, expected=expected)

    try:
        downloaded = bindings.download_snapshot(
            **_snapshot_download_arguments(hub_cache, local_files_only=False)
        )
    except OSError as exc:
        if not _is_windows_no_symlink_error(exc):
            raise
        downloaded = _download_without_windows_symlinks(bindings, expected=expected)
    return _validated_download_path(downloaded, expected=expected)
