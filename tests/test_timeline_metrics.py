from __future__ import annotations

import unittest

from moevm.timeline_metrics import (
    TIMELINE_METRICS_KIND,
    TIMELINE_METRICS_SCHEMA_VERSION,
    CudaInterval,
    interval_from_timestamps,
    summarize_cuda_timeline,
)


class CudaTimelineMetricsTests(unittest.TestCase):
    def test_summarizes_partially_overlapping_transfer_and_compute(self) -> None:
        report = summarize_cuda_timeline(
            transfers=[
                {"name": "copy-2", "start_ms": 8.0, "end_ms": 10.0},
                {"name": "copy-1", "start_ms": 1.0, "end_ms": 5.0},
            ],
            compute=[{"name": "expert-0", "start_ms": 3.0, "end_ms": 9.0}],
        )

        self.assertEqual(report["schema_version"], TIMELINE_METRICS_SCHEMA_VERSION)
        self.assertEqual(report["kind"], TIMELINE_METRICS_KIND)
        self.assertEqual(
            [entry["name"] for entry in report["intervals"]["transfer"]],
            ["copy-1", "copy-2"],
        )
        self.assertEqual(
            report["timeline"],
            {
                "start_ms": 1.0,
                "end_ms": 10.0,
                "span_ms": 9.0,
                "combined_active_duration_ms": 9.0,
            },
        )
        self.assertEqual(report["transfer"]["raw_duration_ms"], 6.0)
        self.assertEqual(report["transfer"]["active_duration_ms"], 6.0)
        self.assertEqual(report["compute"]["active_duration_ms"], 6.0)
        self.assertEqual(
            report["overlap"],
            {
                "duration_ms": 3.0,
                "transfer_overlap_fraction": 0.5,
                "compute_overlap_fraction": 0.5,
                "transfer_hidden_by_compute_ms": 3.0,
                "transfer_exposed_ms": 3.0,
                "compute_exposed_ms": 3.0,
                "serial_active_duration_ms": 12.0,
                "active_duration_saved_by_overlap_ms": 3.0,
            },
        )

    def test_unions_same_lane_before_measuring_cross_lane_overlap(self) -> None:
        report = summarize_cuda_timeline(
            transfers=[
                interval_from_timestamps("copy-a", 0.0, 6.0),
                interval_from_timestamps("copy-b", 2.0, 8.0),
            ],
            compute=[CudaInterval("kernel", 4.0, 10.0)],
        )

        self.assertEqual(report["transfer"]["raw_duration_ms"], 12.0)
        self.assertEqual(report["transfer"]["active_duration_ms"], 8.0)
        self.assertEqual(report["transfer"]["intra_lane_overlap_ms"], 4.0)
        self.assertEqual(report["overlap"]["duration_ms"], 4.0)
        self.assertEqual(report["overlap"]["transfer_overlap_fraction"], 0.5)
        self.assertEqual(report["overlap"]["compute_overlap_fraction"], 2 / 3)
        self.assertEqual(report["overlap"]["serial_active_duration_ms"], 14.0)
        self.assertEqual(report["overlap"]["active_duration_saved_by_overlap_ms"], 4.0)

    def test_lane_end_uses_the_latest_end_not_last_start_ordered_interval(self) -> None:
        report = summarize_cuda_timeline(
            transfers=[
                {"name": "long-copy", "start_ms": 1.0, "end_ms": 10.0},
                {"name": "nested-copy", "start_ms": 3.0, "end_ms": 4.0},
            ],
            compute=[],
        )

        self.assertEqual(report["transfer"]["start_ms"], 1.0)
        self.assertEqual(report["transfer"]["end_ms"], 10.0)

    def test_no_overlap_and_empty_lane_have_unambiguous_zero_or_none_metrics(
        self,
    ) -> None:
        report = summarize_cuda_timeline(
            transfers=[{"name": "copy", "start_ms": 0.0, "end_ms": 2.0}],
            compute=[],
        )

        self.assertEqual(report["compute"]["interval_count"], 0)
        self.assertIsNone(report["compute"]["start_ms"])
        self.assertIsNone(report["compute"]["end_ms"])
        self.assertEqual(report["compute"]["active_duration_ms"], 0.0)
        self.assertEqual(report["overlap"]["duration_ms"], 0.0)
        self.assertEqual(report["overlap"]["transfer_overlap_fraction"], 0.0)
        self.assertIsNone(report["overlap"]["compute_overlap_fraction"])
        self.assertEqual(report["overlap"]["transfer_exposed_ms"], 2.0)

    def test_touching_intervals_do_not_count_as_overlap(self) -> None:
        report = summarize_cuda_timeline(
            transfers=[{"name": "copy", "start_ms": 0.0, "end_ms": 2.0}],
            compute=[{"name": "kernel", "start_ms": 2.0, "end_ms": 5.0}],
        )

        self.assertEqual(report["overlap"]["duration_ms"], 0.0)
        self.assertEqual(report["overlap"]["transfer_overlap_fraction"], 0.0)
        self.assertEqual(report["overlap"]["compute_overlap_fraction"], 0.0)
        self.assertEqual(report["timeline"]["combined_active_duration_ms"], 5.0)

    def test_report_is_deterministic_for_equivalent_input_order(self) -> None:
        transfers = [
            {"name": "b", "start_ms": 4.0, "end_ms": 7.0},
            {"name": "a", "start_ms": 1.0, "end_ms": 5.0},
        ]
        compute = [
            {"name": "compute-b", "start_ms": 5.0, "end_ms": 8.0},
            {"name": "compute-a", "start_ms": 0.0, "end_ms": 2.0},
        ]

        forward = summarize_cuda_timeline(transfers=transfers, compute=compute)
        reverse = summarize_cuda_timeline(
            transfers=list(reversed(transfers)), compute=list(reversed(compute))
        )

        self.assertEqual(forward, reverse)

    def test_zero_length_intervals_are_retained_but_do_not_create_activity(
        self,
    ) -> None:
        report = summarize_cuda_timeline(
            transfers=[{"name": "marker", "start_ms": 4.0, "end_ms": 4.0}],
            compute=[],
        )

        self.assertEqual(report["transfer"]["interval_count"], 1)
        self.assertEqual(report["transfer"]["raw_duration_ms"], 0.0)
        self.assertEqual(report["transfer"]["active_duration_ms"], 0.0)
        self.assertEqual(report["timeline"]["span_ms"], 0.0)
        self.assertIsNone(report["overlap"]["transfer_overlap_fraction"])

    def test_rejects_malformed_or_incomparable_interval_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            CudaInterval("", 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            CudaInterval("copy", float("nan"), 1.0)
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            CudaInterval("copy", 2.0, 1.0)
        with self.assertRaisesRegex(ValueError, "missing required"):
            summarize_cuda_timeline(
                transfers=[{"name": "copy", "start_ms": 0.0}], compute=[]
            )
        with self.assertRaisesRegex(ValueError, "must be iterable"):
            summarize_cuda_timeline(transfers=object(), compute=[])


if __name__ == "__main__":
    unittest.main()
