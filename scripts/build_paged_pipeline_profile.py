#!/usr/bin/env python3
"""Build a hardware-bound sync/async profile from paired benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moevm.pipeline_profile import build_measured_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("SYNC_JSON", "ASYNC_JSON"),
        required=True,
        help="Repeat at least three times; order is always sync then async.",
    )
    parser.add_argument(
        "--minimum-gain",
        type=float,
        default=0.03,
        help="Minimum median sync/async ratio gain required to select async.",
    )
    parser.add_argument("--output", required=True, help="Create-only profile JSON.")
    return parser


def _read_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark result {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark result must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    pairs = [
        (_read_result(Path(sync_path)), _read_result(Path(async_path)))
        for sync_path, async_path in args.pair
    ]
    profile = build_measured_profile(pairs, minimum_gain=args.minimum_gain)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(profile, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output}")
    for pass_name, selected in profile["selection"].items():
        evidence = profile["calibration"]["passes"][pass_name]
        print(
            f"{pass_name}: {selected} "
            f"(median sync/async={evidence['median_sync_over_async']:.6f}x)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
