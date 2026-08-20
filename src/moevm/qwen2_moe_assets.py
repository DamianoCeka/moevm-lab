"""Pinned Qwen2-MoE checkpoint identity and integrity helpers.

The checkpoint remains a third-party asset in the user's Hugging Face cache.
Every required file is pinned to an immutable upstream revision and verified by
both byte size and SHA-256 before it can be used.
"""

from __future__ import annotations

import os
from pathlib import Path

from moevm.pinned_checkpoint import (
    PinnedCheckpointManifest,
    checkpoint_snapshot_path,
    ensure_checkpoint_snapshot,
    verify_checkpoint_snapshot,
)

PINNED_MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
PINNED_REVISION = "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"

PINNED_SHARD_SHA256 = {
    "model-00001-of-00008.safetensors": (
        "44f12f5b3d4e8eebeb55fa733e9dbbd7e119c9bf1502251638223e0dc2a926f0"
    ),
    "model-00002-of-00008.safetensors": (
        "20a99873170c32cd91b489a10ab9408ca73d3f20cf5f57370cf31699aaf2d266"
    ),
    "model-00003-of-00008.safetensors": (
        "e0b7cd920fc98f99e1d83ea856e73f451029ed16dba7ff83b4b839e77ba2d229"
    ),
    "model-00004-of-00008.safetensors": (
        "e2be2af186e5c042e2246947d13640df5bd4ed59512932573996aca155b8e9a9"
    ),
    "model-00005-of-00008.safetensors": (
        "3b75e01d4df9de74823e146a2c1e8826c80f838dcc7d1bd3bf376b68b8a527da"
    ),
    "model-00006-of-00008.safetensors": (
        "c2fc223a88189b3950a4b737fe4c75ae40f792f2e7b30df159afe3cf392589d3"
    ),
    "model-00007-of-00008.safetensors": (
        "88b206c85ee08b9704d2ddb5dcf15b91e5d750884c767c29a93fc91781cb7267"
    ),
    "model-00008-of-00008.safetensors": (
        "996c5057f123db509d3c40c97d6e65dac35eb9ffc71f3ac324a0a531da43e112"
    ),
}
PINNED_SHARD_SIZES = {
    "model-00001-of-00008.safetensors": 3_999_614_664,
    "model-00002-of-00008.safetensors": 3_999_385_424,
    "model-00003-of-00008.safetensors": 3_988_628_968,
    "model-00004-of-00008.safetensors": 3_999_386_088,
    "model-00005-of-00008.safetensors": 3_988_629_648,
    "model-00006-of-00008.safetensors": 3_999_386_104,
    "model-00007-of-00008.safetensors": 3_988_629_648,
    "model-00008-of-00008.safetensors": 668_484_400,
}
PINNED_SHARD_BYTES = sum(PINNED_SHARD_SIZES.values())

_PINNED_METADATA_SHA256 = {
    "LICENSE": "6e521a32ba6661de2b3e7b1cd27d7e1e9b8ff99633c79462a23284cac13c8b8d",
    "README.md": ("793525668ba598ff2c3ff8731ecc548ef212bb46ca918fd6f118b2785861c142"),
    "config.json": ("d0b1cd8f35beccb75211c06940da3274ea986363792ab0ececb60d4ec03dd8c6"),
    "configuration.json": (
        "ab2de9d4e89491b006a99106908418797762e736d95c9f96ab1a8d376f73c458"
    ),
    "generation_config.json": (
        "c5b514da80320749cd57c739c26ddfe44834976109739779a477b60ccde29fb2"
    ),
    "merges.txt": ("599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "model.safetensors.index.json": (
        "ece1b223efe32f4349d0dfa2a522249ac10bcb89369ed25b222c35175cd90b53"
    ),
    "tokenizer.json": (
        "f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8"
    ),
    "tokenizer_config.json": (
        "d30087dd5f3fe386f6d6c029a449798f118ff0afde8824c4d618eb9523d931e6"
    ),
    "vocab.json": ("ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
_PINNED_METADATA_SIZES = {
    "LICENSE": 6_897,
    "README.md": 1_417,
    "config.json": 919,
    "configuration.json": 81,
    "generation_config.json": 144,
    "merges.txt": 1_671_839,
    "model.safetensors.index.json": 416_452,
    "tokenizer.json": 7_028_015,
    "tokenizer_config.json": 1_290,
    "vocab.json": 2_776_833,
}

PINNED_FILE_SHA256 = {**_PINNED_METADATA_SHA256, **PINNED_SHARD_SHA256}
PINNED_FILE_SIZES = {**_PINNED_METADATA_SIZES, **PINNED_SHARD_SIZES}
REQUIRED_SNAPSHOT_FILES = tuple(sorted(PINNED_FILE_SHA256))

MANIFEST = PinnedCheckpointManifest(
    model_id=PINNED_MODEL_ID,
    revision=PINNED_REVISION,
    file_sha256=PINNED_FILE_SHA256,
    file_sizes=PINNED_FILE_SIZES,
)


def pinned_snapshot_path(hf_home: str | os.PathLike[str]) -> Path:
    """Return the canonical Hub-cache path for the pinned Qwen2-MoE revision."""

    return checkpoint_snapshot_path(MANIFEST, hf_home)


def verify_pinned_snapshot(
    snapshot: str | os.PathLike[str],
) -> dict[str, dict[str, int | str]]:
    """Verify every required file in the pinned Qwen2-MoE snapshot."""

    return verify_checkpoint_snapshot(MANIFEST, snapshot)


def ensure_pinned_snapshot(
    hf_home: str | os.PathLike[str], allow_download: bool
) -> Path:
    """Return a verified snapshot, downloading only when explicitly allowed."""

    return ensure_checkpoint_snapshot(
        MANIFEST,
        hf_home,
        allow_download=allow_download,
    )
