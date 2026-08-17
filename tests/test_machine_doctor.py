from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from moevm.config import load_config
from moevm.machine_doctor import (
    _parse_nvidia_smi_rows,
    _parse_proc_meminfo,
    build_memory_ledger,
    collect_machine_report,
    probe_disk,
    probe_nvidia_smi,
    probe_system_memory,
)


class MachineDoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.config = load_config(root / "configs" / "toy.toml")

    def test_memory_ledger_accounts_for_the_complete_model_and_one_token(self) -> None:
        ledger = build_memory_ledger(self.config)
        mib = 1024 * 1024

        self.assertEqual(ledger.expert_bytes, 8 * mib)
        self.assertEqual(ledger.total_experts, 12 * 64)
        self.assertEqual(ledger.vram_cache_slots, 32)
        self.assertEqual(ledger.ram_cache_slots, 256)
        self.assertEqual(ledger.vram_cache_remainder_bytes, 0)
        self.assertEqual(ledger.ram_cache_remainder_bytes, 0)
        self.assertEqual(ledger.all_expert_logical_bytes, 12 * 64 * 8 * mib)
        self.assertEqual(ledger.routed_expert_accesses_per_token, 12 * 4)
        self.assertEqual(ledger.per_layer_logical_routed_expert_bytes, 4 * 8 * mib)
        self.assertEqual(
            ledger.per_token_logical_routed_expert_bytes,
            12 * 4 * 8 * mib,
        )

    def test_proc_meminfo_parser_requires_total_and_available(self) -> None:
        total, available = _parse_proc_meminfo(
            "MemTotal:       1000 kB\nMemAvailable:    250 kB\n"
        )

        self.assertEqual(total, 1000 * 1024)
        self.assertEqual(available, 250 * 1024)
        with self.assertRaisesRegex(ValueError, "MemTotal and MemAvailable"):
            _parse_proc_meminfo("MemTotal:       1000 kB\n")

    def test_system_memory_probe_is_non_fatal(self) -> None:
        memory = probe_system_memory()

        self.assertIn(memory.status, {"available", "unavailable", "error"})
        if memory.status == "available":
            self.assertIsNotNone(memory.total_bytes)
            self.assertIsNotNone(memory.available_bytes)
            self.assertIsNotNone(memory.used_bytes)

    def test_disk_probe_reports_the_volume_for_an_existing_path(self) -> None:
        disk = probe_disk(Path.cwd())

        self.assertEqual(disk.status, "available")
        self.assertIsNotNone(disk.inspected_path)
        self.assertGreater(disk.total_bytes or 0, 0)
        self.assertGreaterEqual(disk.free_bytes or 0, 0)

    def test_disk_probe_uses_a_nearest_existing_ancestor_without_creating_it(
        self,
    ) -> None:
        requested = Path.cwd() / "does-not-exist" / "weights"
        disk = probe_disk(requested)

        self.assertEqual(disk.status, "available")
        self.assertEqual(disk.requested_path, str(requested))
        self.assertNotEqual(disk.inspected_path, str(requested))
        self.assertFalse(requested.exists())

    def test_nvidia_smi_parser_accepts_multiple_gpus(self) -> None:
        gpus = _parse_nvidia_smi_rows(
            "NVIDIA RTX A, 24576, 12345, 555.12\nNVIDIA RTX B, 4096, 512, 555.12\n"
        )

        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[1].index, 1)
        self.assertEqual(gpus[0].total_vram_bytes, 24576 * 1024 * 1024)
        self.assertEqual(gpus[1].free_vram_bytes, 512 * 1024 * 1024)

    @patch("moevm.machine_doctor.subprocess.run")
    def test_nvidia_smi_probe_is_read_only_and_parses_csv(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA RTX A, 24576, 12345, 555.12\n",
            stderr="",
        )

        probe = probe_nvidia_smi(timeout_seconds=0.5)

        self.assertEqual(probe.status, "available")
        self.assertEqual(probe.gpus[0].name, "NVIDIA RTX A")
        self.assertEqual(probe.gpus[0].free_vram_bytes, 12345 * 1024 * 1024)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "nvidia-smi")
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertFalse(run.call_args.kwargs["check"])
        self.assertFalse(run.call_args.kwargs["shell"])

    @patch("moevm.machine_doctor.subprocess.run", side_effect=FileNotFoundError())
    def test_nvidia_smi_missing_is_reported_without_raising(self, _run) -> None:
        probe = probe_nvidia_smi()

        self.assertEqual(probe.status, "unavailable")
        self.assertEqual(probe.gpus, ())

    @patch(
        "moevm.machine_doctor.subprocess.run",
        side_effect=subprocess.TimeoutExpired(["nvidia-smi"], 0.1),
    )
    def test_nvidia_smi_timeout_is_reported_without_raising(self, _run) -> None:
        probe = probe_nvidia_smi(timeout_seconds=0.1)

        self.assertEqual(probe.status, "timeout")
        self.assertEqual(probe.gpus, ())

    @patch("moevm.machine_doctor.subprocess.run")
    def test_nvidia_smi_error_status_is_reported_without_raising(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=9,
            stdout="",
            stderr="driver unavailable",
        )

        probe = probe_nvidia_smi()

        self.assertEqual(probe.status, "error")
        self.assertIn("driver unavailable", probe.detail or "")

    @patch("moevm.machine_doctor.probe_nvidia_smi")
    def test_collection_can_skip_the_optional_gpu_probe(self, gpu_probe) -> None:
        report = collect_machine_report(
            self.config,
            disk_path=Path.cwd(),
            probe_gpu=False,
        )

        self.assertIsNone(report.gpu)
        gpu_probe.assert_not_called()
        self.assertEqual(report.ledger.expert_bytes, 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
