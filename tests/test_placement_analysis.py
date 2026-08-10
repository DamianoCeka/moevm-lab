from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from moevm.cli import main
from moevm.placement_analysis import (
    COLD_LRU,
    HYBRID_HOT_LRU,
    STATIC_HOT,
    PlacementTrace,
    analyze_placement_leave_one_workload_out,
    analyze_placement_train_test,
)
from moevm.trace import write_trace
from moevm.types import RoutingStep


def _trace(
    workload: str,
    source: str,
    token_layers: list[list[tuple[int, ...]]],
    *,
    digest: str | None = None,
) -> PlacementTrace:
    steps = tuple(
        RoutingStep(token, layer, experts)
        for token, layers in enumerate(token_layers)
        for layer, experts in enumerate(layers)
    )
    return PlacementTrace(
        workload_id=workload,
        source=source,
        sha256=digest or hashlib.sha256(source.encode()).hexdigest(),
        steps=steps,
    )


class PlacementAnalysisTests(unittest.TestCase):
    def test_cold_lru_exact_counts_and_per_token_bytes(self) -> None:
        train = (_trace("train", "train", [[(9,)] for _ in range(5)]),)
        test = (
            _trace(
                "test",
                "test",
                [[(0,)], [(1,)], [(0,)], [(2,)], [(0,)]],
            ),
        )

        report = analyze_placement_train_test(
            train,
            test,
            capacity_per_layer=2,
            protected_hot=0,
            expert_bytes=10,
            policies=(COLD_LRU,),
        )

        result = report["evaluations"][COLD_LRU]
        metrics = result["aggregate"]
        self.assertEqual(metrics["accesses"], 5)
        self.assertEqual(metrics["hits"], 2)
        self.assertEqual(metrics["misses"], 3)
        self.assertEqual(metrics["evictions"], 1)
        self.assertEqual(metrics["demand_bytes_loaded"], 30)
        self.assertEqual(metrics["demand_bytes_per_token"], 6.0)
        per_token = result["per_trace"][0]["per_token"]
        self.assertEqual([row["misses"] for row in per_token], [1, 1, 0, 1, 0])
        self.assertEqual(
            [row["demand_bytes_loaded"] for row in per_token], [10, 10, 0, 10, 0]
        )

    def test_per_layer_capacities_are_independent(self) -> None:
        train = (_trace("train", "train", [[(9,), (9,)] for _ in range(3)]),)
        test = (
            _trace(
                "test",
                "test",
                [[(0,), (0,)], [(1,), (1,)], [(0,), (0,)]],
            ),
        )

        report = analyze_placement_train_test(
            train,
            test,
            capacity_per_layer={0: 1, 1: 2},
            protected_hot=0,
            expert_bytes=1,
            policies=(COLD_LRU,),
        )

        per_layer = report["evaluations"][COLD_LRU]["per_trace"][0]["per_layer"]
        self.assertEqual(per_layer[0]["hits"], 0)
        self.assertEqual(per_layer[0]["misses"], 3)
        self.assertEqual(per_layer[0]["evictions"], 2)
        self.assertEqual(per_layer[1]["hits"], 1)
        self.assertEqual(per_layer[1]["misses"], 2)
        self.assertEqual(per_layer[1]["evictions"], 0)

    def test_static_and_hybrid_placements_never_learn_from_test(self) -> None:
        train = (_trace("train", "train", [[(0,)], [(0,)], [(1,)]]),)
        test = (_trace("test", "test", [[(2,)], [(2,)], [(3,)], [(2,)]]),)

        report = analyze_placement_train_test(
            train,
            test,
            capacity_per_layer=2,
            protected_hot=1,
            expert_bytes=10,
            policies=(STATIC_HOT, HYBRID_HOT_LRU),
        )

        static = report["evaluations"][STATIC_HOT]
        self.assertEqual(static["placement"]["hotsets"]["0"], [0, 1])
        self.assertEqual(static["aggregate"]["hits"], 0)
        self.assertEqual(static["aggregate"]["misses"], 4)
        self.assertEqual(static["aggregate"]["evictions"], 0)
        hybrid = report["evaluations"][HYBRID_HOT_LRU]
        self.assertEqual(hybrid["placement"]["hotsets"]["0"], [0])
        self.assertEqual(hybrid["aggregate"]["hits"], 1)
        self.assertEqual(hybrid["aggregate"]["misses"], 3)
        self.assertEqual(hybrid["aggregate"]["evictions"], 2)
        self.assertEqual(hybrid["aggregate"]["preload_bytes"], 10)
        self.assertEqual(hybrid["aggregate"]["total_bytes_loaded"], 40)

    def test_hybrid_keeps_reserved_partition_when_training_has_few_experts(
        self,
    ) -> None:
        train = (_trace("train", "train", [[(0,)], [(0,)]]),)
        test = (_trace("test", "test", [[(1,)], [(2,)], [(1,)]]),)

        report = analyze_placement_train_test(
            train,
            test,
            capacity_per_layer=3,
            protected_hot=2,
            expert_bytes=1,
            policies=(HYBRID_HOT_LRU,),
        )

        hybrid = report["evaluations"][HYBRID_HOT_LRU]
        self.assertEqual(hybrid["placement"]["actual_hotset_slots"]["0"], 1)
        self.assertEqual(hybrid["placement"]["reserved_hot_slots"]["0"], 2)
        self.assertEqual(hybrid["placement"]["dynamic_lru_slots"]["0"], 1)
        per_layer = hybrid["per_trace"][0]["per_layer"][0]
        self.assertEqual(per_layer["unused_reserved_hot_slots"], 1)
        self.assertEqual(hybrid["aggregate"]["hits"], 0)
        self.assertEqual(hybrid["aggregate"]["misses"], 3)
        self.assertEqual(hybrid["aggregate"]["evictions"], 2)

    def test_hotset_ties_use_expert_id_for_reproducibility(self) -> None:
        train = (_trace("train", "train", [[(2,)], [(1,)]]),)
        test = (_trace("test", "test", [[(3,)]]),)

        report = analyze_placement_train_test(
            train,
            test,
            capacity_per_layer=1,
            protected_hot=0,
            expert_bytes=1,
            policies=(STATIC_HOT,),
        )

        hotset = report["evaluations"][STATIC_HOT]["placement"]["hotsets"]["0"]
        self.assertEqual(hotset, [1])

    def test_rejects_train_test_content_overlap(self) -> None:
        digest = "a" * 64
        train = (_trace("train", "train", [[(0,)]], digest=digest),)
        test = (_trace("test", "test", [[(1,)]], digest=digest),)

        with self.assertRaisesRegex(ValueError, "content overlap"):
            analyze_placement_train_test(
                train,
                test,
                capacity_per_layer=1,
                protected_hot=0,
                expert_bytes=1,
            )

    def test_rejects_different_train_test_top_k(self) -> None:
        train = (_trace("train", "train", [[(0,)]]),)
        test = (_trace("test", "test", [[(0, 1)]]),)

        with self.assertRaisesRegex(ValueError, "same top-k"):
            analyze_placement_train_test(
                train,
                test,
                capacity_per_layer=2,
                protected_hot=0,
                expert_bytes=1,
            )

    def test_protected_hot_is_validated_even_without_hybrid_policy(self) -> None:
        train = (_trace("train", "train", [[(0,)]]),)
        test = (_trace("test", "test", [[(1,)]]),)

        with self.assertRaisesRegex(ValueError, "protected_hot"):
            analyze_placement_train_test(
                train,
                test,
                capacity_per_layer=1,
                protected_hot=2,
                expert_bytes=1,
                policies=(COLD_LRU,),
            )

    def test_cross_seed_split_audits_shared_workloads_and_addresses(self) -> None:
        train = (_trace("alpha", "train-alpha", [[(0,)], [(1,)]]),)
        test = (_trace("alpha", "test-alpha", [[(0,)], [(2,)]]),)

        report = analyze_placement_train_test(
            train,
            test,
            capacity_per_layer=1,
            protected_hot=0,
            expert_bytes=1,
            policies=(COLD_LRU,),
        )

        audit = report["split_audit"]
        self.assertEqual(audit["shared_workload_ids"], ["alpha"])
        self.assertFalse(audit["workload_holdout"])
        self.assertEqual(audit["shared_same_address_steps"], 2)
        self.assertEqual(audit["shared_same_address_fraction_of_test"], 1.0)
        self.assertEqual(audit["shared_same_address_exact_expert_set_matches"], 1)

    def test_intra_step_order_is_expert_id_ascending(self) -> None:
        train = (_trace("train", "train", [[(8, 9)], [(8, 9)]]),)
        first_test = (_trace("test", "test-a", [[(2, 1)], [(1, 3)]]),)
        permuted_test = (_trace("test", "test-b", [[(1, 2)], [(3, 1)]]),)
        options = {
            "capacity_per_layer": 1,
            "protected_hot": 0,
            "expert_bytes": 1,
            "policies": (COLD_LRU,),
        }

        first = analyze_placement_train_test(train, first_test, **options)
        permuted = analyze_placement_train_test(train, permuted_test, **options)

        self.assertEqual(
            first["evaluations"][COLD_LRU]["aggregate"],
            permuted["evaluations"][COLD_LRU]["aggregate"],
        )
        self.assertEqual(
            first["parameters"]["intra_step_access_order"], "expert_id_ascending"
        )

    def test_leave_one_workload_out_excludes_every_held_out_trace(self) -> None:
        train = (
            _trace("alpha", "train-alpha", [[(0,)], [(0,)]]),
            _trace("beta", "train-beta", [[(1,)], [(1,)]]),
        )
        test = (
            _trace("alpha", "test-alpha", [[(0,)]]),
            _trace("beta", "test-beta", [[(1,)]]),
        )

        report = analyze_placement_leave_one_workload_out(
            train,
            test,
            capacity_per_layer=1,
            protected_hot=0,
            expert_bytes=4,
            policies=(STATIC_HOT,),
        )

        self.assertEqual(report["protocol"], "leave-one-workload-out")
        for fold in report["folds"]:
            held_out = fold["held_out_workload"]
            trained_workloads = {row["workload_id"] for row in fold["train_manifest"]}
            self.assertNotIn(held_out, trained_workloads)
            self.assertTrue(fold["split_audit"]["workload_holdout"])
            self.assertEqual(fold["split_audit"]["shared_same_address_steps"], 0)
            self.assertEqual(fold["evaluations"][STATIC_HOT]["aggregate"]["misses"], 1)
        aggregate = report["aggregate_across_folds"][STATIC_HOT]
        self.assertEqual(aggregate["tokens"], 2)
        self.assertEqual(aggregate["misses"], 2)

    def test_cli_loads_jsonl_and_writes_report(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            train_path = temporary / "train.trace.jsonl"
            test_path = temporary / "test.trace.jsonl"
            output_path = temporary / "report.json"
            write_trace(
                train_path,
                [RoutingStep(0, layer, (0, 1, 2, 3)) for layer in range(12)],
            )
            write_trace(
                test_path,
                [RoutingStep(0, layer, (1, 2, 3, 4)) for layer in range(12)],
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "analyze-placement",
                        "--config",
                        str(repository / "configs" / "toy.toml"),
                        "--train-trace",
                        str(train_path),
                        "--test-trace",
                        str(test_path),
                        "--capacity-per-layer",
                        "1",
                        "--policy",
                        COLD_LRU,
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("offline placement analysis", stdout.getvalue())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["protocol"], "train-test")
            self.assertEqual(
                payload["evaluations"][COLD_LRU]["aggregate"]["misses"], 48
            )

            stderr = io.StringIO()
            with (
                mock.patch(
                    "moevm.cli.write_placement_analysis",
                    side_effect=OSError("simulated output failure"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                error_code = main(
                    [
                        "analyze-placement",
                        "--config",
                        str(repository / "configs" / "toy.toml"),
                        "--train-trace",
                        str(train_path),
                        "--test-trace",
                        str(test_path),
                        "--capacity-per-layer",
                        "1",
                        "--policy",
                        COLD_LRU,
                        "--output",
                        str(temporary / "failed.json"),
                    ]
                )
            self.assertEqual(error_code, 2)
            self.assertIn("simulated output failure", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
