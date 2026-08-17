from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from moevm import cli
from moevm.machine_doctor import (
    DiskSpace,
    GpuInfo,
    GpuProbe,
    MachineReport,
    SystemMemory,
    build_memory_ledger,
)


class DoctorCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = Path(__file__).resolve().parents[1] / "configs" / "toy.toml"

    def _report(self) -> MachineReport:
        config = cli.load_config(self.config_path)
        return MachineReport(
            ledger=build_memory_ledger(config),
            system_memory=SystemMemory(
                status="available",
                source="fixture",
                total_bytes=64 * 1024**3,
                available_bytes=48 * 1024**3,
            ),
            disk=DiskSpace(
                status="available",
                requested_path="D:\\MoEVM-cache",
                inspected_path="D:\\",
                total_bytes=2 * 1024**4,
                used_bytes=1024**4,
                free_bytes=1024**4,
            ),
            gpu=GpuProbe(
                status="available",
                gpus=(
                    GpuInfo(
                        index=0,
                        name="Fixture GPU",
                        total_vram_bytes=24 * 1024**3,
                        free_vram_bytes=20 * 1024**3,
                        driver_version="555.12",
                    ),
                ),
            ),
        )

    def _main(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_legacy_doctor_output_remains_config_only(self) -> None:
        status, stdout, stderr = self._main(
            ["doctor", "--config", str(self.config_path)]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Configuration is valid.", stdout)
        self.assertNotIn("Machine observations", stdout)

    @patch("moevm.cli.collect_machine_report")
    def test_machine_doctor_renders_observations_and_logical_boundary(
        self, collect
    ) -> None:
        collect.return_value = self._report()

        status, stdout, stderr = self._main(
            [
                "doctor",
                "--config",
                str(self.config_path),
                "--machine",
                "--cache-path",
                "D:\\MoEVM-cache",
            ]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Machine observations", stdout)
        self.assertIn("Fixture GPU", stdout)
        self.assertIn("Configuration memory ledger", stdout)
        self.assertIn("not checkpoint or physical-I/O measurements", stdout)
        self.assertIn("not a claim that the model fits", stdout)
        self.assertEqual(collect.call_args.kwargs["disk_path"], "D:\\MoEVM-cache")
        self.assertTrue(collect.call_args.kwargs["probe_gpu"])

    @patch("moevm.cli.collect_machine_report")
    def test_machine_doctor_json_is_serializable_and_can_skip_gpu(
        self, collect
    ) -> None:
        collect.return_value = self._report()

        status, stdout, stderr = self._main(
            [
                "doctor",
                "--config",
                str(self.config_path),
                "--machine",
                "--no-gpu-probe",
                "--json",
            ]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["ledger"]["vram_cache_slots"], 32)
        self.assertEqual(payload["gpu"]["status"], "available")
        self.assertFalse(collect.call_args.kwargs["probe_gpu"])

    def test_json_requires_machine_mode(self) -> None:
        status, _stdout, stderr = self._main(
            ["doctor", "--config", str(self.config_path), "--json"]
        )

        self.assertEqual(status, 2)
        self.assertIn("--json requires doctor --machine", stderr)


if __name__ == "__main__":
    unittest.main()
