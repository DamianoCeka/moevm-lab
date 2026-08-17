from __future__ import annotations

import builtins
import json
import os
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import patch

from moevm.checkpoint_inspector import inspect_checkpoint

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}
TensorDefinition = tuple[str, str, tuple[int, ...]]


def _elements(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _write_safetensors(
    path: Path, definitions: Iterable[TensorDefinition]
) -> dict[str, int]:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offsets: dict[str, int] = {}
    cursor = 0
    for name, dtype, shape in definitions:
        size = _elements(shape) * _DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + size],
        }
        offsets[name] = size
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(cursor))
    return offsets


def _expert(
    layer: int,
    expert: int,
    *,
    hidden: int,
    intermediate: int,
    dtype: str,
) -> list[TensorDefinition]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return [
        (f"{prefix}.gate_proj.weight", dtype, (intermediate, hidden)),
        (f"{prefix}.up_proj.weight", dtype, (intermediate, hidden)),
        (f"{prefix}.down_proj.weight", dtype, (hidden, intermediate)),
    ]


def _shared(
    layer: int,
    *,
    hidden: int,
    intermediate: int,
    dtype: str,
) -> list[TensorDefinition]:
    prefix = f"model.layers.{layer}.mlp.shared_expert"
    return [
        (f"{prefix}.gate_proj.weight", dtype, (intermediate, hidden)),
        (f"{prefix}.up_proj.weight", dtype, (intermediate, hidden)),
        (f"{prefix}.down_proj.weight", dtype, (hidden, intermediate)),
        (f"model.layers.{layer}.mlp.shared_expert_gate.weight", dtype, (1, hidden)),
    ]


class CheckpointInspectorTests(unittest.TestCase):
    def _qwen_config(self) -> dict[str, object]:
        return {
            "architectures": ["Qwen2MoeForCausalLM"],
            "model_type": "qwen2_moe",
            "hidden_size": 4,
            "intermediate_size": 7,
            "moe_intermediate_size": 3,
            "shared_expert_intermediate_size": 5,
            "num_hidden_layers": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "decoder_sparse_step": 1,
            "hidden_act": "silu",
            "torch_dtype": "bfloat16",
        }

    @staticmethod
    def _write_indexed_snapshot(
        snapshot: Path,
        *,
        config: dict[str, object],
        shards: dict[str, list[TensorDefinition]],
        declared_total: int | None = None,
    ) -> int:
        snapshot.mkdir()
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
        weight_map: dict[str, str] = {}
        logical_bytes = 0
        for filename, definitions in shards.items():
            sizes = _write_safetensors(snapshot / filename, definitions)
            logical_bytes += sum(sizes.values())
            weight_map.update({name: filename for name in sizes})
        metadata: dict[str, object] = {}
        if declared_total is not None:
            metadata["total_size"] = declared_total
        else:
            metadata["total_size"] = logical_bytes
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": metadata, "weight_map": weight_map}),
            encoding="utf-8",
        )
        return logical_bytes

    @staticmethod
    def _write_config_guard_snapshot(snapshot: Path, config: dict[str, object]) -> None:
        """Write an invalid shard so config guards must fail before parsing it."""

        snapshot.mkdir()
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"not-a-safetensors-file")

    def _symlink_or_skip(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target)
        except OSError as exc:  # pragma: no cover - platform capability
            self.skipTest(f"symbolic links are unavailable: {exc}")

    def test_inspects_qwen2_moe_shared_experts_and_shard_placement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            layer_zero_expert_zero = _expert(
                0, 0, hidden=4, intermediate=3, dtype="BF16"
            )
            layer_zero_expert_one = _expert(
                0, 1, hidden=4, intermediate=3, dtype="BF16"
            )
            layer_one_expert_zero = _expert(
                1, 0, hidden=4, intermediate=3, dtype="BF16"
            )
            layer_one_expert_one = _expert(1, 1, hidden=4, intermediate=3, dtype="BF16")
            unrelated = ("model.layers.1.self_attn.q_proj.weight", "BF16", (4, 4))
            # L0/E1 crosses shards. L1/E0 is colocated but an unrelated tensor
            # creates a physical gap between its projections.
            shards = {
                "model-00001-of-00002.safetensors": [
                    ("model.embed_tokens.weight", "BF16", (8, 4)),
                    *layer_zero_expert_zero,
                    layer_zero_expert_one[0],
                    *_shared(0, hidden=4, intermediate=5, dtype="BF16"),
                ],
                "model-00002-of-00002.safetensors": [
                    *layer_zero_expert_one[1:],
                    layer_one_expert_zero[0],
                    unrelated,
                    *layer_one_expert_zero[1:],
                    *layer_one_expert_one,
                    *_shared(1, hidden=4, intermediate=5, dtype="BF16"),
                ],
            }
            logical_bytes = self._write_indexed_snapshot(
                snapshot,
                config=self._qwen_config(),
                shards=shards,
            )

            before = {
                path.relative_to(snapshot): path.read_bytes()
                for path in snapshot.iterdir()
            }
            real_import = builtins.__import__

            def reject_torch(name: str, *args: object, **kwargs: object):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("the inspector must not import torch")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=reject_torch):
                report = inspect_checkpoint(snapshot)
            after = {
                path.relative_to(snapshot): path.read_bytes()
                for path in snapshot.iterdir()
            }

            self.assertEqual(before, after)
            self.assertTrue(report.read_only)
            self.assertFalse(report.network_used)
            self.assertFalse(report.model_code_executed)
            self.assertEqual(report.model.model_type, "qwen2_moe")
            self.assertEqual(report.model.expert_intermediate_size, 3)
            self.assertEqual(report.experts.naming, "gate_up_down")
            self.assertEqual(report.experts.layers, (0, 1))
            self.assertEqual(report.experts.total_experts, 4)
            self.assertEqual(report.experts.tensor_count, 12)
            self.assertEqual(report.experts.bytes_per_expert, 72)
            self.assertEqual(report.experts.logical_bytes, 288)
            self.assertEqual(report.experts.colocated_experts, 3)
            self.assertEqual(report.experts.contiguous_experts, 2)
            self.assertEqual(report.experts.split_experts, 1)

            placements = {
                (placement.layer, placement.expert): placement
                for placement in report.experts.placements
            }
            self.assertFalse(placements[(0, 1)].colocated)
            self.assertIsNone(placements[(0, 1)].span_bytes)
            self.assertTrue(placements[(1, 0)].colocated)
            self.assertFalse(placements[(1, 0)].contiguous)
            self.assertEqual(placements[(1, 0)].gap_bytes, 32)
            self.assertEqual(
                placements[(0, 0)].projection_order,
                ("gate_proj", "up_proj", "down_proj"),
            )

            self.assertTrue(report.shared_expert.present)
            self.assertEqual(report.shared_expert.layers, (0, 1))
            self.assertEqual(report.shared_expert.intermediate_size, 5)
            self.assertEqual(report.shared_expert.bytes_per_layer, 120)
            self.assertEqual(report.shared_expert.logical_bytes, 240)
            self.assertEqual(report.shared_expert.gate_logical_bytes, 16)
            self.assertEqual(report.logical_tensor_bytes, logical_bytes)
            self.assertEqual(report.declared_total_size_bytes, logical_bytes)
            self.assertEqual(len(report.shards), 2)
            self.assertEqual(
                sum(shard.expert_tensor_count for shard in report.shards), 12
            )
            self.assertEqual(
                sum(shard.shared_expert_tensor_count for shard in report.shards),
                8,
            )
            json.dumps(report.to_dict(), allow_nan=False, sort_keys=True)

    def test_inspects_unsharded_legacy_mixtral_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            config = {
                "architectures": ["MixtralForCausalLM"],
                "model_type": "mixtral",
                "hidden_size": 4,
                "intermediate_size": 3,
                "num_hidden_layers": 1,
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
                "hidden_act": "silu",
                "torch_dtype": "float16",
            }
            (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
            definitions: list[TensorDefinition] = []
            for expert in range(2):
                prefix = f"model.layers.0.block_sparse_moe.experts.{expert}"
                definitions.extend(
                    [
                        (f"{prefix}.w1.weight", "F16", (3, 4)),
                        (f"{prefix}.w3.weight", "F16", (3, 4)),
                        (f"{prefix}.w2.weight", "F16", (4, 3)),
                    ]
                )
            _write_safetensors(snapshot / "model.safetensors", definitions)
            (snapshot / "sentencepiece_ja.py").write_text(
                "raise RuntimeError('must not run')\n", encoding="utf-8"
            )

            report = inspect_checkpoint(snapshot / "model.safetensors")

            self.assertEqual(report.index_kind, "unsharded_synthesized")
            self.assertIsNone(report.index_path)
            self.assertEqual(report.experts.naming, "mixtral_w1_w2_w3")
            self.assertEqual(report.experts.total_experts, 2)
            self.assertEqual(report.experts.bytes_per_expert, 72)
            self.assertFalse(report.shared_expert.present)
            self.assertTrue(any("unsharded" in warning for warning in report.warnings))
            self.assertTrue(
                any("sentencepiece_ja.py" in warning for warning in report.warnings)
            )

    def test_rejects_missing_expert_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            shards = {
                "model.safetensors": [
                    *_expert(0, 0, hidden=4, intermediate=3, dtype="BF16"),
                    *_expert(0, 1, hidden=4, intermediate=3, dtype="BF16"),
                    *_shared(0, hidden=4, intermediate=5, dtype="BF16"),
                ]
            }
            self._write_indexed_snapshot(
                snapshot,
                config=self._qwen_config(),
                shards=shards,
            )

            with self.assertRaisesRegex(ValueError, "expert layer coverage"):
                inspect_checkpoint(snapshot)

    def test_rejects_shared_expert_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            definitions: list[TensorDefinition] = []
            for layer in range(2):
                definitions.extend(
                    _expert(layer, 0, hidden=4, intermediate=3, dtype="BF16")
                )
                definitions.extend(
                    _expert(layer, 1, hidden=4, intermediate=3, dtype="BF16")
                )
                definitions.extend(
                    _shared(layer, hidden=4, intermediate=5, dtype="BF16")
                )
            wrong_name = "model.layers.1.mlp.shared_expert.up_proj.weight"
            definitions = [
                (name, dtype, (6, 4) if name == wrong_name else shape)
                for name, dtype, shape in definitions
            ]
            self._write_indexed_snapshot(
                snapshot,
                config=self._qwen_config(),
                shards={"model.safetensors": definitions},
            )

            with self.assertRaisesRegex(ValueError, "shared-expert shape mismatch"):
                inspect_checkpoint(snapshot)

    def test_rejects_index_path_escape_and_windows_ads(self) -> None:
        unsafe_names = ("../outside.safetensors", "C:/outside.safetensors", "x:ads")
        for unsafe in unsafe_names:
            with (
                self.subTest(unsafe=unsafe),
                tempfile.TemporaryDirectory() as temporary,
            ):
                snapshot = Path(temporary) / "snapshot"
                snapshot.mkdir()
                (snapshot / "config.json").write_text(
                    json.dumps(self._qwen_config()), encoding="utf-8"
                )
                (snapshot / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"total_size": 1},
                            "weight_map": {"tensor": unsafe},
                        }
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "escapes snapshot"):
                    inspect_checkpoint(snapshot)

    def test_rejects_index_header_assignment_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            actual = "model.layers.0.mlp.experts.0.gate_proj.weight"
            _write_safetensors(
                snapshot / "model.safetensors", [(actual, "BF16", (3, 4))]
            )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 24},
                        "weight_map": {
                            actual.replace("gate", "up"): "model.safetensors"
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "absent from declared shard"):
                inspect_checkpoint(snapshot)

    def test_rejects_noncontiguous_safetensors_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
            header = {
                tensor_name: {
                    "dtype": "BF16",
                    "shape": [3, 4],
                    "data_offsets": [2, 26],
                }
            }
            encoded = json.dumps(header).encode("utf-8")
            encoded += b" " * ((-len(encoded)) % 8)
            (snapshot / "model.safetensors").write_bytes(
                len(encoded).to_bytes(8, "little") + encoded + bytes(26)
            )

            with self.assertRaisesRegex(ValueError, "offsets are not contiguous"):
                inspect_checkpoint(snapshot)

    def test_rejects_huge_layer_and_expert_counts_before_shard_parsing(self) -> None:
        for key in ("num_hidden_layers", "num_experts"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                snapshot = Path(temporary) / "snapshot"
                config = self._qwen_config()
                config[key] = 10**12
                self._write_config_guard_snapshot(snapshot, config)

                with self.assertRaisesRegex(
                    ValueError, rf"config {key} exceeds inspector safety limit"
                ):
                    inspect_checkpoint(snapshot)

    def test_rejects_oversized_combined_sparse_expert_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            config = self._qwen_config()
            config["num_hidden_layers"] = 4096
            config["num_experts"] = 16_384
            self._write_config_guard_snapshot(snapshot, config)

            with self.assertRaisesRegex(
                ValueError,
                "sparse layer/expert inventory exceeds inspector safety limit",
            ):
                inspect_checkpoint(snapshot)

    def test_rejects_boolean_config_counts(self) -> None:
        for key in ("num_hidden_layers", "num_experts", "decoder_sparse_step"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                snapshot = Path(temporary) / "snapshot"
                config = self._qwen_config()
                config[key] = True
                self._write_config_guard_snapshot(snapshot, config)

                with self.assertRaisesRegex(
                    ValueError, rf"config {key} must be a positive integer"
                ):
                    inspect_checkpoint(snapshot)

    def test_rejects_invalid_layer_and_expert_counts(self) -> None:
        for key in ("num_hidden_layers", "num_experts"):
            for value in (0, -1, 1.5, "2"):
                with (
                    self.subTest(key=key, value=value),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    snapshot = Path(temporary) / "snapshot"
                    config = self._qwen_config()
                    config[key] = value
                    self._write_config_guard_snapshot(snapshot, config)

                    with self.assertRaisesRegex(
                        ValueError, rf"config {key} must be a positive integer"
                    ):
                        inspect_checkpoint(snapshot)

    def test_rejects_weight_map_above_tensor_entry_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "tensor.0": "missing.safetensors",
                            "tensor.1": "missing.safetensors",
                            "tensor.2": "missing.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("moevm.checkpoint_inspector._MAX_TENSORS", 2),
                self.assertRaisesRegex(
                    ValueError, "weight_map exceeds.*tensor entries"
                ),
            ):
                inspect_checkpoint(snapshot)

    def test_rejects_weight_map_above_shard_limit_before_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "tensor.0": "missing-0.safetensors",
                            "tensor.1": "missing-1.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("moevm.checkpoint_inspector._MAX_SHARDS", 1),
                self.assertRaisesRegex(ValueError, "weight_map exceeds.*shards"),
            ):
                inspect_checkpoint(snapshot)

    def test_rejects_aggregate_header_bytes_before_reading_next_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            filenames = ("model-1.safetensors", "model-2.safetensors")
            names = ("tensor.0", "tensor.1")
            header_sizes: list[int] = []
            for filename, name in zip(filenames, names, strict=True):
                shard_path = snapshot / filename
                _write_safetensors(shard_path, [(name, "BF16", (1,))])
                header_sizes.append(
                    int.from_bytes(shard_path.read_bytes()[:8], "little")
                )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 4},
                        "weight_map": dict(zip(names, filenames, strict=True)),
                    }
                ),
                encoding="utf-8",
            )
            aggregate_limit = sum(header_sizes) - 1
            self.assertGreaterEqual(aggregate_limit, max(header_sizes))

            with (
                patch(
                    "moevm.checkpoint_inspector._MAX_AGGREGATE_HEADER_BYTES",
                    aggregate_limit,
                ),
                self.assertRaisesRegex(ValueError, "aggregate safetensors header"),
            ):
                inspect_checkpoint(snapshot)

    def test_rejects_aggregate_header_tensor_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            filename = "model.safetensors"
            _write_safetensors(
                snapshot / filename,
                [("tensor.0", "BF16", (1,)), ("tensor.1", "BF16", (1,))],
            )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"tensor.0": filename}}), encoding="utf-8"
            )

            with (
                patch("moevm.checkpoint_inspector._MAX_TENSORS", 1),
                self.assertRaisesRegex(ValueError, "aggregate safetensors tensor"),
            ):
                inspect_checkpoint(snapshot)

    def test_rejects_physical_shard_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            original = snapshot / "model-original.safetensors"
            alias = snapshot / "model-alias.safetensors"
            _write_safetensors(original, [("tensor.0", "BF16", (1,))])
            try:
                os.link(original, alias)
            except OSError as exc:  # pragma: no cover - platform capability
                self.skipTest(f"hard links are unavailable: {exc}")
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "tensor.0": original.name,
                            "tensor.1": alias.name,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "aliases one physical shard"):
                inspect_checkpoint(snapshot)

    def test_rejects_shard_symlink_that_escapes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(self._qwen_config()), encoding="utf-8"
            )
            outside = root / "outside.safetensors"
            _write_safetensors(outside, [("tensor.0", "BF16", (1,))])
            link = snapshot / "linked.safetensors"
            self._symlink_or_skip(link, outside)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"tensor.0": link.name}}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "escapes allowed checkpoint roots"):
                inspect_checkpoint(snapshot)

    def test_accepts_canonical_huggingface_blob_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "models--Qwen--fixture"
            blobs = repository / "blobs"
            snapshot = repository / "snapshots" / "pinned-revision"
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)

            definitions: list[TensorDefinition] = []
            for layer in range(2):
                for expert in range(2):
                    definitions.extend(
                        _expert(layer, expert, hidden=4, intermediate=3, dtype="BF16")
                    )
                definitions.extend(
                    _shared(layer, hidden=4, intermediate=5, dtype="BF16")
                )
            shard_blob = blobs / "shard-blob"
            sizes = _write_safetensors(shard_blob, definitions)
            config_blob = blobs / "config-blob"
            config_blob.write_text(json.dumps(self._qwen_config()), encoding="utf-8")
            index_blob = blobs / "index-blob"
            index_blob.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": sum(sizes.values())},
                        "weight_map": {name: "model.safetensors" for name in sizes},
                    }
                ),
                encoding="utf-8",
            )
            relative_blob_root = Path("..") / ".." / "blobs"
            for link_name, blob_name in (
                ("config.json", config_blob.name),
                ("model.safetensors.index.json", index_blob.name),
                ("model.safetensors", shard_blob.name),
            ):
                self._symlink_or_skip(
                    snapshot / link_name, relative_blob_root / blob_name
                )

            report = inspect_checkpoint(snapshot)

            self.assertEqual(report.experts.total_experts, 4)
            self.assertTrue(report.shared_expert.present)
            self.assertEqual(report.logical_tensor_bytes, sum(sizes.values()))

    def test_rejects_default_metadata_symlinks_outside_allowed_roots(self) -> None:
        for artifact in ("config.json", "model.safetensors.index.json"):
            with (
                self.subTest(artifact=artifact),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                outside = root / f"outside-{artifact.replace('.', '-')}"
                if artifact == "config.json":
                    outside.write_text(
                        json.dumps(self._qwen_config()), encoding="utf-8"
                    )
                    (snapshot / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": {"tensor.0": "missing.safetensors"}}),
                        encoding="utf-8",
                    )
                else:
                    outside.write_text(
                        json.dumps({"weight_map": {"tensor.0": "missing.safetensors"}}),
                        encoding="utf-8",
                    )
                    (snapshot / "config.json").write_text(
                        json.dumps(self._qwen_config()), encoding="utf-8"
                    )
                self._symlink_or_skip(snapshot / artifact, outside)

                with self.assertRaisesRegex(
                    ValueError, "escapes allowed checkpoint roots"
                ):
                    inspect_checkpoint(snapshot)


if __name__ == "__main__":
    unittest.main()
