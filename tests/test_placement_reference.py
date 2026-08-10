from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from moevm.placement_analysis import (
    PlacementTrace,
    analyze_placement_leave_one_workload_out,
    analyze_placement_train_test,
    load_placement_traces,
    write_placement_analysis,
)


def _load_canonical_traces(root: Path, seed: int) -> tuple[PlacementTrace, ...]:
    trace_root = (
        root
        / "benchmarks"
        / "reference"
        / "real-routing-olmoe-m1"
        / "traces"
        / f"seed-{seed}"
    )
    paths = tuple(sorted(trace_root.glob("*.trace.jsonl")))
    loaded = load_placement_traces(paths)
    return tuple(
        PlacementTrace(
            workload_id=trace.workload_id,
            source=path.relative_to(root).as_posix(),
            sha256=trace.sha256,
            steps=trace.steps,
        )
        for path, trace in zip(paths, loaded, strict=True)
    )


class PlacementReferenceTests(unittest.TestCase):
    def test_real_trace_reports_match_compact_reference(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference_path = (
            root
            / "benchmarks"
            / "reference"
            / "real-routing-olmoe-m1"
            / "placement"
            / "summary.json"
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        self.assertEqual(
            reference["full_report_serialization"],
            {
                "allow_nan": False,
                "encoding": "UTF-8",
                "final_newline": False,
                "indent": 2,
                "newline": "LF",
                "sort_keys": True,
            },
        )
        parameters = reference["parameters"]
        train = _load_canonical_traces(root, seed=17)
        test = _load_canonical_traces(root, seed=29)
        options = {
            "capacity_per_layer": {
                int(layer): slots
                for layer, slots in parameters["capacity_per_layer"].items()
            },
            "protected_hot": parameters["hybrid_protected_hot_per_layer"],
            "expert_bytes": parameters["expert_bytes"],
            "policies": tuple(parameters["policies"]),
        }

        cross_seed = analyze_placement_train_test(train, test, **options)
        leave_one_out = analyze_placement_leave_one_workload_out(
            train,
            test,
            **options,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            cross_payload = write_placement_analysis(
                temporary / "cross-seed.json",
                cross_seed,
            ).read_bytes()
            leave_one_out_payload = write_placement_analysis(
                temporary / "leave-one-out.json",
                leave_one_out,
            ).read_bytes()
        for payload in (cross_payload, leave_one_out_payload):
            self.assertNotIn(b"\r\n", payload)
            self.assertFalse(payload.endswith(b"\n"))

        manifest = reference["trace_manifest"]
        self.assertEqual(cross_seed["train_manifest"], manifest["seed_17_training"])
        self.assertEqual(cross_seed["test_manifest"], manifest["seed_29_test"])
        self.assertEqual(
            leave_one_out["source_train_manifest"], manifest["seed_17_training"]
        )
        self.assertEqual(
            leave_one_out["source_test_manifest"], manifest["seed_29_test"]
        )
        self.assertEqual(cross_seed["parameters"], parameters)
        self.assertEqual(leave_one_out["parameters"], parameters)

        cross_reference = reference["reports"]["cross_seed_same_workload"]
        cross_aggregates = {
            policy: evaluation["aggregate"]
            for policy, evaluation in cross_seed["evaluations"].items()
        }
        placement_digests = {
            policy: evaluation["placement"]["sha256"]
            for policy, evaluation in cross_seed["evaluations"].items()
        }
        self.assertEqual(cross_seed["split_audit"], cross_reference["split_audit"])
        self.assertEqual(cross_aggregates, cross_reference["aggregate"])
        self.assertEqual(
            placement_digests,
            cross_reference["placement_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(cross_payload).hexdigest(),
            cross_reference["full_report_sha256"],
        )

        leave_one_out_reference = reference["reports"]["leave_one_workload_out"]
        fold_audits = [
            {
                "held_out_workload": fold["held_out_workload"],
                "split_audit": fold["split_audit"],
            }
            for fold in leave_one_out["folds"]
        ]
        self.assertEqual(
            leave_one_out["split_audit"],
            leave_one_out_reference["split_audit"],
        )
        self.assertEqual(
            fold_audits,
            leave_one_out_reference["fold_split_audits"],
        )
        self.assertEqual(
            leave_one_out["aggregate_across_folds"],
            leave_one_out_reference["aggregate"],
        )
        self.assertEqual(
            hashlib.sha256(leave_one_out_payload).hexdigest(),
            leave_one_out_reference["full_report_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
