from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from moevm import cli
from moevm.checkpoint_inspector import (
    CheckpointInspection,
    ExpertInspection,
    ModelInspection,
    ShardInspection,
    SharedExpertInspection,
)


class CheckpointInspectorCliTests(unittest.TestCase):
    @staticmethod
    def _report(*, warnings: tuple[str, ...] = ()) -> CheckpointInspection:
        return CheckpointInspection(
            schema="moevm.checkpoint-inspection",
            schema_version=1,
            snapshot_path="D:\\models\\fixture",
            config_path="D:\\models\\fixture\\config.json",
            index_path=("D:\\models\\fixture\\model.safetensors.index.json"),
            index_kind="indexed",
            read_only=True,
            network_used=False,
            model_code_executed=False,
            model=ModelInspection(
                model_type="qwen2_moe",
                architectures=("Qwen2MoeForCausalLM",),
                hidden_size=4,
                dense_intermediate_size=7,
                expert_intermediate_size=3,
                shared_expert_intermediate_size=5,
                hidden_layers=2,
                expected_expert_layers=(0, 1),
                experts_per_layer=2,
                experts_per_token=1,
                hidden_act="silu",
                declared_dtype="BF16",
            ),
            experts=ExpertInspection(
                naming="gate_up_down",
                layers=(0, 1),
                projections=("gate_proj", "up_proj", "down_proj"),
                dtype="BF16",
                hidden_size=4,
                intermediate_size=3,
                experts_per_layer=2,
                total_experts=4,
                tensor_count=12,
                bytes_per_expert=72,
                logical_bytes=288,
                colocated_experts=3,
                contiguous_experts=2,
                split_experts=1,
                placements=(),
            ),
            shared_expert=SharedExpertInspection(
                present=True,
                layers=(0, 1),
                dtype="BF16",
                hidden_size=4,
                intermediate_size=5,
                projection_tensor_count=6,
                gate_tensor_count=2,
                bytes_per_layer=120,
                logical_bytes=240,
                gate_logical_bytes=16,
            ),
            shards=(
                ShardInspection(
                    filename="model-00001-of-00002.safetensors",
                    file_size_bytes=1_024,
                    header_size_bytes=512,
                    tensor_count=10,
                    logical_bytes=272,
                    expert_tensor_count=6,
                    shared_expert_tensor_count=4,
                ),
                ShardInspection(
                    filename="model-00002-of-00002.safetensors",
                    file_size_bytes=1_024,
                    header_size_bytes=512,
                    tensor_count=10,
                    logical_bytes=272,
                    expert_tensor_count=6,
                    shared_expert_tensor_count=4,
                ),
            ),
            tensor_count=20,
            logical_tensor_bytes=544,
            declared_total_size_bytes=544,
            warnings=warnings,
        )

    @staticmethod
    def _main(argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    @patch("moevm.cli.inspect_checkpoint")
    def test_console_summary_contains_shape_placement_safety_and_warnings(
        self, inspect
    ) -> None:
        inspect.return_value = self._report(
            warnings=("Executable or pickle-capable files were ignored: model.py",)
        )

        status, stdout, stderr = self._main(
            [
                "inspect-checkpoint",
                "D:\\models\\fixture",
                "--config",
                "D:\\configs\\fixture.json",
            ]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Qwen2MoeForCausalLM", stdout)
        self.assertIn("type: qwen2_moe", stdout)
        self.assertIn("Layers: 2 total; 2 sparse", stdout)
        self.assertIn("Experts: 2 per layer; 4 inspected; top-1", stdout)
        self.assertIn("Expert bytes: 72 bytes each; 288 bytes logical total", stdout)
        self.assertIn("3/4 colocated; 2/4 contiguous; 1 split", stdout)
        self.assertIn("Shared expert: present on 2 layers", stdout)
        self.assertIn("Read-only: yes; network used: no", stdout)
        self.assertIn("Warnings:", stdout)
        self.assertIn("model.py", stdout)
        inspect.assert_called_once_with(
            "D:\\models\\fixture",
            config_path="D:\\configs\\fixture.json",
        )

    @patch("moevm.cli.inspect_checkpoint")
    def test_json_uses_checkpoint_report_to_dict(self, inspect) -> None:
        report = self._report()
        inspect.return_value = report

        status, stdout, stderr = self._main(
            ["inspect-checkpoint", "D:\\models\\fixture", "--json"]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        expected = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(json.loads(stdout), expected)
        inspect.assert_called_once_with("D:\\models\\fixture", config_path=None)

    @patch(
        "moevm.cli.inspect_checkpoint",
        side_effect=FileNotFoundError("checkpoint not found"),
    )
    def test_inspection_errors_use_existing_cli_error_contract(self, _inspect) -> None:
        status, stdout, stderr = self._main(["inspect-checkpoint", "D:\\missing"])

        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("error: checkpoint not found", stderr)


if __name__ == "__main__":
    unittest.main()
