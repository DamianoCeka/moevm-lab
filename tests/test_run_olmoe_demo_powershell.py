from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "run_olmoe_demo.ps1"
_CMD = _ROOT / "demo.cmd"


@unittest.skipUnless(sys.platform == "win32", "requires Windows PowerShell")
class OlmoeDemoPowerShellTests(unittest.TestCase):
    def test_script_parses_without_powershell_errors(self) -> None:
        escaped_script = str(_SCRIPT).replace("'", "''")
        command = (
            "$tokens=$null; $errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{escaped_script}', "
            "[ref]$tokens, [ref]$errors) | Out-Null; "
            "if ($errors.Count -ne 0) { $errors | Out-String | Write-Error; exit 1 }"
        )
        completed = subprocess.run(
            ("powershell.exe", "-NoLogo", "-NoProfile", "-Command", command),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dry_run_with_spaced_paths_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="MoEVM demo ") as temporary:
            root = Path(temporary)
            cache = root / "cache with spaces"
            output = root / "results with spaces"
            before = tuple(root.iterdir())

            completed = subprocess.run(
                (
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(_SCRIPT),
                    "-DryRun",
                    "-PythonPath",
                    sys.executable,
                    "-CachePath",
                    str(cache),
                    "-OutputRoot",
                    str(output),
                ),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Dry run only", completed.stdout)
            self.assertEqual(tuple(root.iterdir()), before)

    def test_cmd_is_a_thin_argument_forwarder(self) -> None:
        raw = _CMD.read_text(encoding="utf-8").lower()

        self.assertIn("run_olmoe_demo.ps1", raw)
        self.assertIn("%*", raw)
        self.assertIn("exit /b %errorlevel%", raw)

    def test_wrapper_does_not_auto_execute_drive_discovered_python(self) -> None:
        raw = _SCRIPT.read_text(encoding="utf-8").lower()

        self.assertNotIn("moevm-lab-envs", raw)

    def test_offline_missing_snapshot_fails_without_creating_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="MoEVM offline ") as temporary:
            root = Path(temporary)
            cache = root / "missing cache"
            output = root / "missing output"

            completed = subprocess.run(
                (
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(_SCRIPT),
                    "-Offline",
                    "-CachePath",
                    str(cache),
                    "-OutputRoot",
                    str(output),
                ),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Offline mode was requested", completed.stderr)
            self.assertEqual(tuple(root.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
