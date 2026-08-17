from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as torch_functional
    from safetensors.torch import save_file
except ImportError:
    torch = None

if torch is not None:
    from moevm.paged_runtime import (
        CachePolicy,
        ExpertLoadTicket,
        ExpertSlotCache,
        PagedExpertRuntime,
        SafetensorExpertStore,
        TransformersMoEAdapter,
        _cuda_timeline_payload,
        _CudaTimelineCapture,
        _CudaTimelineEventSpan,
        attach_transformers_moe_runtime,
        attach_transformers_olmoe_runtime,
        load_non_expert_weights_into_meta_model,
        register_transformers_paged_experts,
        transformers_moe_adapter_for_model,
        transformers_paged_experts_forward,
        validate_transformers_paged_model,
    )
    from moevm.types import ExpertKey


@unittest.skipUnless(torch is not None, "requires the real-traces dependencies")
class PagedRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.snapshot = Path(self.temporary_directory.name)
        generator = torch.Generator().manual_seed(20260810)
        self.hidden_size = 4
        self.intermediate_size = 3
        self.original: dict[ExpertKey, tuple[torch.Tensor, ...]] = {}
        shards: list[dict[str, torch.Tensor]] = [{}, {}]
        weight_map: dict[str, str] = {}

        for layer in range(2):
            for expert in range(4):
                key = ExpertKey(layer, expert)
                gate = torch.randn(
                    self.intermediate_size, self.hidden_size, generator=generator
                )
                up = torch.randn(
                    self.intermediate_size, self.hidden_size, generator=generator
                )
                down = torch.randn(
                    self.hidden_size, self.intermediate_size, generator=generator
                )
                self.original[key] = (gate, up, down)
                for projection, tensor in zip(
                    ("gate_proj", "up_proj", "down_proj"),
                    (gate, up, down),
                    strict=True,
                ):
                    tensor_name = (
                        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    )
                    shard_index = (layer + expert + (projection == "down_proj")) % 2
                    shards[shard_index][tensor_name] = tensor
                    weight_map[tensor_name] = f"model-{shard_index + 1}.safetensors"

        self.non_expert = torch.randn(2, 2, generator=generator).to(torch.bfloat16)
        shards[0]["other.weight"] = self.non_expert
        weight_map["other.weight"] = "model-1.safetensors"
        for shard_index, tensors in enumerate(shards, start=1):
            save_file(tensors, self.snapshot / f"model-{shard_index}.safetensors")
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        self.store = SafetensorExpertStore(self.snapshot)
        self.addCleanup(self.store.close)

    def _cache(
        self,
        capacity: int,
        *,
        policy: CachePolicy | str = "lru",
        static_keys: tuple[ExpertKey, ...] = (),
        device: str = "cpu",
        staging_slots: int = 1,
        pipeline_mode: str = "sync",
    ) -> ExpertSlotCache:
        return ExpertSlotCache(
            self.store,
            capacity=capacity,
            device=device,
            policy=policy,
            static_keys=static_keys,
            staging_slots=staging_slots,
            pin_staging=device == "cuda",
            pipeline_mode=pipeline_mode,
        )

    def _reference_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert in torch.unique(top_k_index, sorted=True).tolist():
            gate_weight, up_weight, down_weight = self.original[ExpertKey(0, expert)]
            gate_weight = gate_weight.to(hidden_states.device)
            up_weight = up_weight.to(hidden_states.device)
            down_weight = down_weight.to(hidden_states.device)
            token_index, top_k_position = torch.where(top_k_index == expert)
            states = hidden_states[token_index]
            gate = torch_functional.linear(states, gate_weight)
            up = torch_functional.linear(states, up_weight)
            states = torch_functional.silu(gate) * up
            states = torch_functional.linear(states, down_weight)
            states *= top_k_weights[token_index, top_k_position, None]
            output.index_add_(0, token_index, states)
        return output

    def test_cuda_timeline_payload_uses_one_origin_and_retains_expert_labels(
        self,
    ) -> None:
        class FakeEvent:
            def __init__(self, timestamp_ms: float) -> None:
                self.timestamp_ms = timestamp_ms

            @staticmethod
            def query() -> bool:
                return True

        class FakeOrigin:
            @staticmethod
            def elapsed_time(event: FakeEvent) -> float:
                return event.timestamp_ms

        payload = _cuda_timeline_payload(
            FakeOrigin(),
            [
                _CudaTimelineEventSpan(
                    lane="h2d",
                    key=ExpertKey(0, 1),
                    sequence=0,
                    started_event=FakeEvent(1.0),
                    ended_event=FakeEvent(5.0),
                ),
                _CudaTimelineEventSpan(
                    lane="expert_compute",
                    key=ExpertKey(0, 2),
                    sequence=1,
                    started_event=FakeEvent(3.0),
                    ended_event=FakeEvent(8.0),
                ),
            ],
        )

        self.assertEqual(payload["status"], "measured")
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["unit"], "milliseconds")
        self.assertEqual(payload["spans"][0]["name"], "h2d:0:L0:E1")
        self.assertEqual(payload["spans"][1]["name"], "expert_compute:1:L0:E2")
        self.assertEqual(payload["summary"]["overlap"]["duration_ms"], 2.0)

    def test_cuda_timeline_close_waits_for_worker_commit_and_seals_spans(self) -> None:
        class FakeEvent:
            next_timestamp_ms = 0.0

            def __init__(self, **_kwargs: object) -> None:
                self.timestamp_ms = FakeEvent.next_timestamp_ms
                FakeEvent.next_timestamp_ms += 1.0

            def record(self, _stream: object) -> None:
                return None

            def query(self) -> bool:
                return True

            def elapsed_time(self, event: FakeEvent) -> float:
                return event.timestamp_ms - self.timestamp_ms

        class FakeStream:
            def __init__(self) -> None:
                self.waited_for: list[FakeEvent] = []

            def wait_event(self, event: FakeEvent) -> None:
                self.waited_for.append(event)

        with (
            patch("moevm.paged_runtime.torch.cuda.Event", FakeEvent),
            patch("moevm.paged_runtime.torch.cuda.synchronize"),
        ):
            capture = _CudaTimelineCapture(torch.device("cuda"))
            stream = FakeStream()
            capture.begin(stream)
            ticket = ExpertLoadTicket(
                key=ExpertKey(0, 0),
                queued_at=time.perf_counter(),
                request_clock=0,
            )
            self.assertTrue(capture.reserve_transfer(ticket))

            claimed = threading.Event()
            release_worker = threading.Event()
            finished = threading.Event()
            worker_errors: list[Exception] = []
            result: list[dict[str, object]] = []

            def worker() -> None:
                try:
                    self.assertTrue(capture.claim_transfer(ticket))
                    claimed.set()
                    if not release_worker.wait(timeout=2.0):
                        raise TimeoutError("test did not release CUDA worker")
                    capture.wait_on_origin(stream)
                    started = FakeEvent()
                    ended = FakeEvent()
                    self.assertTrue(capture.commit_transfer(ticket, started, ended))
                except Exception as exc:  # noqa: BLE001 - asserted below
                    worker_errors.append(exc)

            def close_capture() -> None:
                try:
                    result.append(capture.finish())
                except Exception as exc:  # noqa: BLE001 - asserted below
                    worker_errors.append(exc)
                finally:
                    finished.set()

            worker_thread = threading.Thread(target=worker)
            closer_thread = threading.Thread(target=close_capture)
            worker_thread.start()
            self.assertTrue(claimed.wait(timeout=1.0))
            closer_thread.start()
            try:
                self.assertFalse(finished.wait(timeout=0.1))
            finally:
                release_worker.set()
                worker_thread.join(timeout=2.0)
                closer_thread.join(timeout=2.0)

            self.assertFalse(worker_thread.is_alive())
            self.assertFalse(closer_thread.is_alive())
            self.assertFalse(worker_errors)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]["complete"])
            self.assertEqual(len(result[0]["spans"]), 1)
            self.assertEqual(len(stream.waited_for), 1)

            # Capture close freezes the append-only trace.  A worker that did
            # not claim before close cannot add a later H2D span.
            late_ticket = ExpertLoadTicket(
                key=ExpertKey(0, 1),
                queued_at=time.perf_counter(),
                request_clock=0,
            )
            self.assertFalse(capture.claim_transfer(late_ticket))
            self.assertEqual(len(capture._spans), 1)

    def test_cuda_timeline_cancelled_worker_reservation_is_incomplete(self) -> None:
        class FakeEvent:
            def __init__(self, **_kwargs: object) -> None:
                self.timestamp_ms = 0.0

            def record(self, _stream: object) -> None:
                return None

            def query(self) -> bool:
                return True

            def elapsed_time(self, event: FakeEvent) -> float:
                return event.timestamp_ms - self.timestamp_ms

        class FakeStream:
            def wait_event(self, _event: FakeEvent) -> None:
                return None

        with (
            patch("moevm.paged_runtime.torch.cuda.Event", FakeEvent),
            patch("moevm.paged_runtime.torch.cuda.synchronize"),
        ):
            capture = _CudaTimelineCapture(torch.device("cuda"))
            capture.begin(FakeStream())
            ticket = ExpertLoadTicket(
                key=ExpertKey(0, 0),
                queued_at=time.perf_counter(),
                request_clock=0,
            )
            self.assertTrue(capture.reserve_transfer(ticket))

            result = capture.finish()

        self.assertFalse(result["complete"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["schema_version"], 1)
        self.assertIn("cancelled", str(result["reason"]))
        self.assertEqual(result["spans"], [])

    def test_cuda_timeline_coverage_mismatch_is_incomplete(self) -> None:
        class FakeEvent:
            next_timestamp_ms = 0.0

            def __init__(self, **_kwargs: object) -> None:
                self.timestamp_ms = FakeEvent.next_timestamp_ms
                FakeEvent.next_timestamp_ms += 1.0

            def record(self, _stream: object) -> None:
                return None

            def query(self) -> bool:
                return True

            def elapsed_time(self, event: FakeEvent) -> float:
                return event.timestamp_ms - self.timestamp_ms

        class FakeStream:
            def wait_event(self, _event: FakeEvent) -> None:
                return None

        with (
            patch("moevm.paged_runtime.torch.cuda.Event", FakeEvent),
            patch("moevm.paged_runtime.torch.cuda.synchronize"),
        ):
            capture = _CudaTimelineCapture(
                torch.device("cuda"),
                transfer_loads_baseline=10,
            )
            stream = FakeStream()
            capture.begin(stream)
            capture.record_transfer(ExpertKey(0, 0), FakeEvent(), FakeEvent())
            result = capture.finish(transfer_loads_after_synchronize=lambda: 12)

        self.assertFalse(result["complete"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["schema_version"], 2)
        self.assertIn("coverage mismatch", str(result["reason"]))
        self.assertEqual(result["spans"], [])
        self.assertEqual(
            result["coverage"],
            {"cache_transfer_loads_delta": 2, "h2d_span_count": 1},
        )

    def test_covered_cuda_timeline_abort_is_v2(self) -> None:
        capture = _CudaTimelineCapture(
            torch.device("cuda"),
            transfer_loads_baseline=0,
        )
        capture._state = capture._ACTIVE

        capture.abort()

        self.assertIsNotNone(capture.result)
        self.assertFalse(capture.result["complete"])
        self.assertEqual(capture.result["status"], "incomplete")
        self.assertEqual(capture.result["schema_version"], 2)

    def test_cuda_timeline_runtime_coverage_scopes_drain_and_end_poll(self) -> None:
        class FakeEvent:
            next_timestamp_ms = 0.0

            def __init__(self, **_kwargs: object) -> None:
                self.timestamp_ms = FakeEvent.next_timestamp_ms
                FakeEvent.next_timestamp_ms += 1.0

            def record(self, _stream: object) -> None:
                return None

            def query(self) -> bool:
                return True

            def elapsed_time(self, event: FakeEvent) -> float:
                return event.timestamp_ms - self.timestamp_ms

        class FakeStream:
            def wait_event(self, _event: FakeEvent) -> None:
                return None

        class FakeMetrics:
            def __init__(self, transfer_loads: int) -> None:
                self.transfer_loads = transfer_loads

        class FakeCache:
            def __init__(self) -> None:
                self.device = torch.device("cuda")
                self.execution_lock = threading.RLock()
                self.transfer_loads = 10
                self.operations: list[str] = []

            def wait_idle(self) -> None:
                self.operations.append("wait_idle")
                # The pre-origin drain retires two earlier H2Ds, which must
                # not appear in this capture's scoped delta.
                self.transfer_loads += 2

            def _poll_transfer_completions(self) -> None:
                self.operations.append("poll")
                # The capture's committed H2D is credited only after the
                # device synchronization at close.
                self.transfer_loads += 1

            def metrics(self) -> FakeMetrics:
                self.operations.append("metrics")
                return FakeMetrics(self.transfer_loads)

        cache = FakeCache()
        runtime = PagedExpertRuntime(cache)  # type: ignore[arg-type]
        stream = FakeStream()

        def synchronize(_device: torch.device) -> None:
            cache.operations.append("synchronize")

        with (
            patch("moevm.paged_runtime.torch.cuda.Event", FakeEvent),
            patch(
                "moevm.paged_runtime.torch.cuda.current_stream",
                return_value=stream,
            ),
            patch(
                "moevm.paged_runtime.torch.cuda.synchronize",
                side_effect=synchronize,
            ),
        ):
            capture = runtime._begin_cuda_timeline_capture()
            capture.record_transfer(ExpertKey(0, 0), FakeEvent(), FakeEvent())
            runtime._end_cuda_timeline_capture(capture, failed=False)

        self.assertIsNotNone(capture.result)
        self.assertTrue(capture.result["complete"])
        self.assertEqual(capture.result["status"], "not_applicable")
        self.assertEqual(capture.result["schema_version"], 2)
        self.assertEqual(
            capture.result["coverage"],
            {"cache_transfer_loads_delta": 1, "h2d_span_count": 1},
        )
        self.assertEqual(
            cache.operations,
            ["wait_idle", "metrics", "synchronize", "poll", "metrics"],
        )

    def test_cuda_timeline_runtime_coverage_reports_empty_complete_scope(self) -> None:
        class FakeEvent:
            next_timestamp_ms = 0.0

            def __init__(self, **_kwargs: object) -> None:
                self.timestamp_ms = FakeEvent.next_timestamp_ms
                FakeEvent.next_timestamp_ms += 1.0

            def record(self, _stream: object) -> None:
                return None

            def query(self) -> bool:
                return True

            def elapsed_time(self, event: FakeEvent) -> float:
                return event.timestamp_ms - self.timestamp_ms

        class FakeStream:
            def wait_event(self, _event: FakeEvent) -> None:
                return None

        class FakeMetrics:
            transfer_loads = 7

        class FakeCache:
            device = torch.device("cuda")
            execution_lock = threading.RLock()

            @staticmethod
            def wait_idle() -> None:
                return None

            @staticmethod
            def _poll_transfer_completions() -> None:
                return None

            @staticmethod
            def metrics() -> FakeMetrics:
                return FakeMetrics()

        runtime = PagedExpertRuntime(FakeCache())  # type: ignore[arg-type]
        stream = FakeStream()
        with (
            patch("moevm.paged_runtime.torch.cuda.Event", FakeEvent),
            patch(
                "moevm.paged_runtime.torch.cuda.current_stream",
                return_value=stream,
            ),
            patch("moevm.paged_runtime.torch.cuda.synchronize"),
            runtime.cuda_timeline_capture() as capture,
        ):
            pass

        self.assertIsNotNone(capture.result)
        self.assertTrue(capture.result["complete"])
        self.assertEqual(capture.result["status"], "not_applicable")
        self.assertEqual(capture.result["schema_version"], 2)
        self.assertEqual(
            capture.result["coverage"],
            {"cache_transfer_loads_delta": 0, "h2d_span_count": 0},
        )

    def test_cuda_timeline_ledger_retains_each_reserved_ticket_identity(self) -> None:
        """Layer-local tickets cannot have their object id reused mid-capture."""

        capture = _CudaTimelineCapture(torch.device("cuda"))
        capture._state = capture._ACTIVE
        first = ExpertLoadTicket(
            key=ExpertKey(0, 0),
            queued_at=time.perf_counter(),
            request_clock=0,
        )
        second = ExpertLoadTicket(
            key=ExpertKey(1, 0),
            queued_at=time.perf_counter(),
            request_clock=0,
        )

        self.assertTrue(capture.reserve_transfer(first))
        self.assertTrue(capture.reserve_transfer(second))

        self.assertEqual(len(capture._transfer_ledger), 2)
        self.assertIs(capture._transfer_ledger[id(first)].ticket, first)
        self.assertIs(capture._transfer_ledger[id(second)].ticket, second)

    def test_shared_transfer_stream_submissions_do_not_interleave(self) -> None:
        """A worker and foreground H2D keep complete event pairs contiguous."""

        class FakeEvent:
            def __init__(self, **_kwargs: object) -> None:
                return None

            def record(self, _stream: object) -> None:
                return None

        class FakeStream:
            def wait_event(self, _event: FakeEvent) -> None:
                return None

            def synchronize(self) -> None:
                return None

        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        submissions: list[str] = []

        class FakeStreamScope:
            def __enter__(self) -> None:
                name = threading.current_thread().name
                submissions.append(name)
                if name == "first-transfer":
                    first_entered.set()
                    if not release_first.wait(timeout=2.0):
                        raise TimeoutError("test did not release first H2D submission")
                else:
                    second_entered.set()

            def __exit__(self, *_args: object) -> bool:
                return False

        cache = self._cache(2, staging_slots=2, pipeline_mode="async")
        self.addCleanup(cache.close)
        cache._transfer_stream = FakeStream()

        def submit(slot: int) -> None:
            cache._enqueue_staging_to_slot(
                slot,
                cache._staging_gate_up[slot],
                cache._staging_down[slot],
                (),
            )

        with (
            patch("moevm.paged_runtime.torch.cuda.Event", FakeEvent),
            patch(
                "moevm.paged_runtime.torch.cuda.stream",
                side_effect=lambda _stream: FakeStreamScope(),
            ),
        ):
            first = threading.Thread(target=submit, args=(0,), name="first-transfer")
            second = threading.Thread(target=submit, args=(1,), name="second-transfer")
            first.start()
            self.assertTrue(first_entered.wait(timeout=1.0))
            second.start()
            try:
                self.assertFalse(second_entered.wait(timeout=0.1))
            finally:
                release_first.set()
                first.join(timeout=2.0)
                second.join(timeout=2.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(submissions, ["first-transfer", "second-transfer"])

    def test_cuda_timeline_scope_serializes_one_runtime_for_its_full_lifetime(
        self,
    ) -> None:
        runtime = PagedExpertRuntime(self._cache(2))
        capture = object()
        runtime._begin_cuda_timeline_capture = lambda: capture  # type: ignore[method-assign]
        runtime._end_cuda_timeline_capture = (  # type: ignore[method-assign]
            lambda _capture, *, failed: None
        )
        entered = threading.Event()
        release = threading.Event()
        contender_entered = threading.Event()
        failures: list[Exception] = []

        def hold_capture() -> None:
            try:
                with runtime.cuda_timeline_capture():
                    entered.set()
                    if not release.wait(timeout=2.0):
                        raise TimeoutError("test did not release timeline scope")
            except (
                RuntimeError,
                TimeoutError,
            ) as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        def contend_for_runtime() -> None:
            with runtime._forward_lock:
                contender_entered.set()

        holder = threading.Thread(target=hold_capture)
        contender = threading.Thread(target=contend_for_runtime)
        holder.start()
        try:
            self.assertTrue(entered.wait(timeout=1.0))
            contender.start()
            self.assertFalse(contender_entered.wait(timeout=0.1))
        finally:
            release.set()
            holder.join(timeout=2.0)
            contender.join(timeout=2.0)

        self.assertFalse(holder.is_alive())
        self.assertFalse(contender.is_alive())
        self.assertFalse(failures)
        self.assertTrue(contender_entered.is_set())

    def test_store_indexes_and_packs_a_cross_shard_expert(self) -> None:
        key = ExpertKey(0, 1)

        loaded = self.store.load(key)
        gate, up, down = self.original[key]

        self.assertEqual(len(self.store), 8)
        self.assertEqual(self.store.layers, (0, 1))
        self.assertEqual(self.store.experts_in_layer(0), (0, 1, 2, 3))
        torch.testing.assert_close(loaded.gate_up, torch.cat((gate, up)))
        torch.testing.assert_close(loaded.down, down)
        self.assertEqual(
            self.store.spec.size_bytes, (gate.numel() * 2 + down.numel()) * 4
        )

    def test_store_accepts_hugging_face_snapshot_symlink_layout(self) -> None:
        cache_root = self.snapshot / "hf-cache"
        blobs = cache_root / "blobs"
        snapshot = cache_root / "snapshots" / "revision"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)

        filenames = (
            "model.safetensors.index.json",
            "model-1.safetensors",
            "model-2.safetensors",
        )
        for index, filename in enumerate(filenames):
            blob = blobs / f"content-addressed-{index}"
            blob.write_bytes((self.snapshot / filename).read_bytes())
            try:
                (snapshot / filename).symlink_to(blob)
            except OSError as exc:
                self.skipTest(f"filesystem symlinks unavailable: {exc}")

        with SafetensorExpertStore(snapshot) as linked_store:
            loaded = linked_store.load(ExpertKey(0, 1))

        gate, up, down = self.original[ExpertKey(0, 1)]
        torch.testing.assert_close(loaded.gate_up, torch.cat((gate, up)))
        torch.testing.assert_close(loaded.down, down)

    def test_lru_policy_and_allocations_are_bounded(self) -> None:
        cache = self._cache(2)
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        two = ExpertKey(0, 2)

        cache.get(zero)
        cache.get(one)
        cache.get(zero)
        cache.get(two)

        self.assertEqual(cache.resident_keys, (zero, two))
        self.assertEqual(cache.allocated_cache_bytes, 2 * self.store.spec.size_bytes)
        self.assertEqual(cache.allocated_staging_bytes, self.store.spec.size_bytes)
        metrics = cache.metrics()
        self.assertEqual(metrics.requests, 4)
        self.assertEqual(metrics.hits, 1)
        self.assertEqual(metrics.misses, 3)
        self.assertEqual(metrics.evictions, 1)
        self.assertEqual(metrics.storage_bytes, 3 * self.store.spec.size_bytes)
        self.assertEqual(metrics.host_to_device_bytes, 0)

    def test_storage_failure_preserves_the_previous_victim(self) -> None:
        cache = self._cache(1)
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        cache.get(zero)
        original_load_into = self.store.load_into

        def fail_read(*_args, **_kwargs):
            raise OSError("injected read failure")

        self.store.load_into = fail_read
        try:
            with self.assertRaisesRegex(OSError, "injected read failure"):
                cache.get(one)
        finally:
            self.store.load_into = original_load_into

        self.assertEqual(cache.resident_keys, (zero,))
        cache.get(zero)
        self.assertEqual(cache.metrics().hits, 1)

    def test_transfer_failure_leaves_slot_empty_and_recovers(self) -> None:
        cache = self._cache(1)
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        cache.get(zero)
        original_copy = cache._copy_staging_to_slot

        def fail_copy(*_args, **_kwargs):
            raise RuntimeError("injected transfer failure")

        cache._copy_staging_to_slot = fail_copy
        try:
            with self.assertRaisesRegex(RuntimeError, "injected transfer failure"):
                cache.get(one)
        finally:
            cache._copy_staging_to_slot = original_copy

        self.assertEqual(cache.resident_keys, ())
        cache.get(one)
        self.assertEqual(cache.resident_keys, (one,))
        self.assertEqual(cache.metrics().evictions, 1)

    def test_async_cpu_coalesces_load_and_keeps_metrics_after_close(self) -> None:
        cache = self._cache(
            1,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        key = ExpertKey(0, 0)
        original_load_into = self.store.load_into
        load_count = 0
        load_lock = threading.Lock()

        def counted_load(*args, **kwargs):
            nonlocal load_count
            with load_lock:
                load_count += 1
            return original_load_into(*args, **kwargs)

        self.store.load_into = counted_load
        try:
            first = cache.submit(key)
            second = cache.submit(key)
            self.assertIs(first, second)
            loaded = cache.resolve(first)
        finally:
            self.store.load_into = original_load_into

        gate, up, down = self.original[key]
        torch.testing.assert_close(loaded.gate_up, torch.cat((gate, up)))
        torch.testing.assert_close(loaded.down, down)
        self.assertEqual(load_count, 1)
        cache.wait_idle()
        cache.close()
        cache.close()
        metrics = cache.metrics()
        self.assertEqual(metrics.requests, 2)
        self.assertEqual(metrics.hits, 0)
        self.assertEqual(metrics.misses, 2)
        self.assertEqual(metrics.coalesced_requests, 1)
        self.assertEqual(metrics.storage_loads, 1)
        self.assertEqual(metrics.transfer_loads, 1)
        self.assertEqual(metrics.storage_failures, 0)
        self.assertEqual(metrics.storage_bytes, self.store.spec.size_bytes)
        self.assertLessEqual(metrics.peak_staging_in_use, 2)
        self.assertEqual(
            self.store.load(key).gate_up.shape, self.store.spec.gate_up_shape
        )

    def test_async_cpu_forward_is_bounded_and_matches_reference(self) -> None:
        generator = torch.Generator().manual_seed(808)
        hidden_states = torch.randn(5, self.hidden_size, generator=generator)
        top_k_index = torch.tensor(
            [[0, 1], [2, 1], [3, 0], [1, 3], [2, 0]], dtype=torch.long
        )
        top_k_weights = torch.rand(5, 2, generator=generator)
        cache = self._cache(
            1,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)

        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        actual = runtime.forward(
            0,
            hidden_states,
            top_k_index,
            top_k_weights,
        )

        torch.testing.assert_close(actual, expected)
        cache.wait_idle()
        metrics = cache.metrics()
        self.assertEqual(metrics.requests, 4)
        self.assertEqual(metrics.misses, 4)
        self.assertLessEqual(metrics.pending_loads_peak, 2)
        self.assertLessEqual(metrics.peak_staging_in_use, 2)
        self.assertEqual(cache._stage_state, ["free", "free"])

    def test_async_wait_idle_discards_undemanded_lookahead_without_admission(
        self,
    ) -> None:
        cache = self._cache(
            1,
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        key = ExpertKey(0, 0)

        ticket = cache._submit_lookahead(key)
        self.assertTrue(ticket.storage_done.wait(timeout=2.0))
        self.assertEqual(cache.resident_keys, ())
        self.assertEqual(cache._key_to_slot, {})
        self.assertEqual(cache._slot_to_key, [None])

        cache.wait_idle()

        self.assertEqual(ticket.state, "discarded")
        self.assertEqual(cache.resident_keys, ())
        self.assertEqual(cache._pending_by_key, {})
        self.assertEqual(cache._stage_state, ["free"])
        self.assertEqual(cache._slot_reservation, [None])
        metrics = cache.metrics()
        self.assertEqual((metrics.requests, metrics.hits, metrics.misses), (0, 0, 0))
        self.assertEqual(metrics.evictions, 0)

    def test_async_wait_idle_waits_for_worker_acknowledgement_of_queued_discard(
        self,
    ) -> None:
        cache = self._cache(
            1,
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        worker_started = threading.Event()
        release_worker = threading.Event()
        self.addCleanup(release_worker.set)
        original_worker = cache._io_worker

        def blocked_worker() -> None:
            worker_started.set()
            if not release_worker.wait(timeout=2.0):
                raise TimeoutError("test did not release the I/O worker")
            original_worker()

        cache._io_worker = blocked_worker
        ticket = cache._submit_lookahead(ExpertKey(0, 0))
        self.assertTrue(worker_started.wait(timeout=2.0))
        errors: list[Exception] = []

        def drain() -> None:
            try:
                cache.wait_idle()
            except Exception as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)

        drainer = threading.Thread(target=drain)
        drainer.start()
        deadline = time.monotonic() + 2.0
        while not ticket.discard_requested and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(ticket.discard_requested)
        self.assertTrue(drainer.is_alive())
        self.assertIs(cache._pending_by_key[ticket.key], ticket)
        self.assertEqual(cache._queued_job_count, 1)

        release_worker.set()
        drainer.join(timeout=2.0)
        self.assertFalse(drainer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(ticket.state, "discarded")
        self.assertEqual(cache._pending_by_key, {})
        self.assertEqual(cache._queued_job_count, 0)

    def test_async_cpu_lookahead_never_evicts_a_logical_resident(self) -> None:
        cache = self._cache(
            1,
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        resident = ExpertKey(0, 0)
        future = ExpertKey(0, 1)

        cache.get(resident)
        ticket = cache._submit_lookahead(future)
        self.assertTrue(ticket.storage_done.wait(timeout=2.0))

        self.assertEqual(cache.resident_keys, (resident,))
        self.assertIsNone(ticket.destination_slot)
        self.assertEqual(cache._slot_reservation, [None])
        self.assertEqual(cache.metrics().evictions, 0)

        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (resident,))
        self.assertEqual(cache.metrics().evictions, 0)

    def test_adaptive_fills_empty_partition_async_then_uses_sync(self) -> None:
        cache = self._cache(
            2,
            staging_slots=2,
            pipeline_mode="adaptive",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        generator = torch.Generator().manual_seed(1508)

        first_hidden = torch.randn(3, self.hidden_size, generator=generator)
        first_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
        first_weights = torch.ones(3, 1)
        first = runtime.forward(0, first_hidden, first_index, first_weights)
        torch.testing.assert_close(
            first,
            self._reference_forward(first_hidden, first_index, first_weights),
        )

        second_hidden = torch.randn(2, self.hidden_size, generator=generator)
        second_index = torch.tensor([[1], [2]], dtype=torch.long)
        second_weights = torch.ones(2, 1)
        second = runtime.forward(0, second_hidden, second_index, second_weights)
        torch.testing.assert_close(
            second,
            self._reference_forward(second_hidden, second_index, second_weights),
        )

        metrics = cache.metrics()
        self.assertEqual(metrics.adaptive_async_forwards, 1)
        self.assertEqual(metrics.adaptive_sync_forwards, 1)
        self.assertEqual(metrics.adaptive_async_experts, 3)
        self.assertEqual(metrics.adaptive_sync_experts, 2)
        self.assertEqual((metrics.requests, metrics.hits, metrics.misses), (5, 2, 3))
        self.assertEqual(cache._stage_state, ["free", "free"])

    def test_adaptive_uses_sync_for_only_one_missing_expert(self) -> None:
        cache = self._cache(
            2,
            staging_slots=2,
            pipeline_mode="adaptive",
        )
        self.addCleanup(cache.close)
        cache.get(ExpertKey(0, 0))
        hidden_states = torch.ones(2, self.hidden_size)
        top_k_index = torch.tensor([[0], [1]], dtype=torch.long)
        top_k_weights = torch.ones(2, 1)

        actual = PagedExpertRuntime(cache).forward(
            0,
            hidden_states,
            top_k_index,
            top_k_weights,
        )

        torch.testing.assert_close(
            actual,
            self._reference_forward(hidden_states, top_k_index, top_k_weights),
        )
        metrics = cache.metrics()
        self.assertEqual(metrics.adaptive_async_forwards, 0)
        self.assertEqual(metrics.adaptive_sync_forwards, 1)
        self.assertEqual(metrics.adaptive_sync_experts, 2)
        self.assertIsNone(cache._worker)

    def test_adaptive_single_staging_slot_remains_synchronous(self) -> None:
        cache = self._cache(2, pipeline_mode="adaptive")
        self.addCleanup(cache.close)
        hidden_states = torch.ones(3, self.hidden_size)
        top_k_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
        top_k_weights = torch.ones(3, 1)

        actual = PagedExpertRuntime(cache).forward(
            0,
            hidden_states,
            top_k_index,
            top_k_weights,
        )

        torch.testing.assert_close(
            actual,
            self._reference_forward(hidden_states, top_k_index, top_k_weights),
        )
        metrics = cache.metrics()
        self.assertEqual(metrics.adaptive_async_forwards, 0)
        self.assertEqual(metrics.adaptive_sync_forwards, 1)
        self.assertIsNone(cache._worker)

    def test_adaptive_rejects_public_async_ticket_api(self) -> None:
        cache = self._cache(
            2,
            staging_slots=2,
            pipeline_mode="adaptive",
        )
        self.addCleanup(cache.close)

        with self.assertRaisesRegex(RuntimeError, "pipeline_mode='async'"):
            cache.submit(ExpertKey(0, 0))

        loaded = cache.get(ExpertKey(0, 0))
        gate, up, down = self.original[ExpertKey(0, 0)]
        torch.testing.assert_close(loaded.gate_up, torch.cat((gate, up)))
        torch.testing.assert_close(loaded.down, down)
        self.assertIsNone(cache._worker)

    def test_async_capable_cache_switches_pipeline_at_drained_boundary(self) -> None:
        cache = self._cache(
            2,
            staging_slots=2,
            pipeline_mode="async",
        )
        runtime = PagedExpertRuntime(cache)
        self.addCleanup(cache.close)

        cache.set_pipeline_mode("sync")
        first_hidden = torch.ones(2, self.hidden_size)
        first_index = torch.tensor([[0], [1]], dtype=torch.long)
        first_weights = torch.ones(2, 1)
        first = runtime.forward(0, first_hidden, first_index, first_weights)
        torch.testing.assert_close(
            first,
            self._reference_forward(first_hidden, first_index, first_weights),
        )
        self.assertIsNone(cache._worker)

        cache.set_pipeline_mode("async")
        second_index = torch.tensor([[2], [3]], dtype=torch.long)
        second = runtime.forward(0, first_hidden, second_index, first_weights)
        torch.testing.assert_close(
            second,
            self._reference_forward(first_hidden, second_index, first_weights),
        )
        worker = cache._worker
        self.assertIsNotNone(worker)

        cache.set_pipeline_mode("sync")
        cache.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual((cache.metrics().requests, cache.metrics().misses), (4, 4))

    def test_sync_only_cache_cannot_enable_async_later(self) -> None:
        cache = self._cache(2)
        self.addCleanup(cache.close)

        with self.assertRaisesRegex(RuntimeError, "async infrastructure"):
            cache.set_pipeline_mode("async")
        with self.assertRaisesRegex(ValueError, "pipeline_mode"):
            cache.set_pipeline_mode("automatic")

    def test_async_lookahead_matches_sync_demand_and_lru_accounting(self) -> None:
        sync_cache = self._cache(2)
        async_cache = self._cache(
            2,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(async_cache.close)
        for key in (ExpertKey(0, 0), ExpertKey(0, 2)):
            sync_cache.get(key)
            async_cache.get(key)

        sync_before = sync_cache.metrics()
        async_before = async_cache.metrics()
        generator = torch.Generator().manual_seed(1808)
        hidden_states = torch.randn(2, self.hidden_size, generator=generator)
        top_k_index = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
        top_k_weights = torch.rand(2, 2, generator=generator)

        sync_output = PagedExpertRuntime(sync_cache).forward(
            0,
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        async_output = PagedExpertRuntime(async_cache).forward(
            0,
            hidden_states,
            top_k_index,
            top_k_weights,
        )

        torch.testing.assert_close(async_output, sync_output)
        sync_after = sync_cache.metrics()
        async_after = async_cache.metrics()
        fields = (
            "requests",
            "hits",
            "misses",
            "evictions",
            "storage_loads",
            "transfer_loads",
            "storage_bytes",
            "host_to_device_bytes",
        )
        for field_name in fields:
            sync_delta = getattr(sync_after, field_name) - getattr(
                sync_before, field_name
            )
            async_delta = getattr(async_after, field_name) - getattr(
                async_before, field_name
            )
            self.assertEqual(async_delta, sync_delta, field_name)
        self.assertEqual(
            (
                async_after.requests - async_before.requests,
                async_after.hits - async_before.hits,
                async_after.misses - async_before.misses,
                async_after.evictions - async_before.evictions,
            ),
            (3, 1, 2, 2),
        )
        self.assertEqual(async_cache.resident_keys, sync_cache.resident_keys)

    def test_async_pipeline_schedules_future_read_during_current_compute(self) -> None:
        hidden_states = torch.ones(3, self.hidden_size)
        top_k_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
        top_k_weights = torch.ones(3, 1)
        cache = self._cache(
            2,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )

        compute_started = threading.Event()
        future_read_started = threading.Event()
        original_load_into = self.store.load_into
        original_linear = torch_functional.linear
        linear_calls = 0

        def coordinated_load(key, *args, **kwargs):
            if key == ExpertKey(0, 2):
                future_read_started.set()
                if not compute_started.wait(timeout=2):
                    raise AssertionError(
                        "compute did not start while the future read was active"
                    )
            return original_load_into(key, *args, **kwargs)

        def coordinated_linear(*args, **kwargs):
            nonlocal linear_calls
            linear_calls += 1
            if linear_calls == 1:
                compute_started.set()
                if not future_read_started.wait(timeout=2):
                    raise AssertionError("future read did not overlap scheduling")
            return original_linear(*args, **kwargs)

        self.store.load_into = coordinated_load
        torch_functional.linear = coordinated_linear
        try:
            actual = runtime.forward(
                0,
                hidden_states,
                top_k_index,
                top_k_weights,
            )
        finally:
            torch_functional.linear = original_linear
            self.store.load_into = original_load_into

        torch.testing.assert_close(actual, expected)
        self.assertTrue(compute_started.is_set())
        self.assertTrue(future_read_started.is_set())

    def test_async_metrics_separate_reader_queue_wait_from_storage_queue(self) -> None:
        """Reader backlog is measured independently of the legacy aggregate."""

        cache = self._cache(
            1,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        ticket = ExpertLoadTicket(
            key=ExpertKey(0, 0),
            queued_at=0.0,
            request_clock=0,
        )
        jobs = cache._jobs
        self.assertIsNotNone(jobs)

        with patch(
            "moevm.paged_runtime.time.perf_counter",
            side_effect=(10.0, 13.0, 20.0, 23.0),
        ):
            cache._enqueue_ticket(ticket)
            jobs.put_nowait(None)
            cache._io_worker()

        self.assertIsNone(ticket.reader_queue_enqueued_at)
        self.assertTrue(ticket.storage_done.is_set())
        metrics = cache.metrics()
        self.assertEqual(metrics.reader_queue_wait_seconds, 3.0)
        # Kept for compatibility: this legacy value still includes all time
        # from submit construction through staging acquisition.
        self.assertEqual(metrics.storage_queue_seconds, 20.0)
        self.assertEqual(metrics.staging_wait_seconds, 0.0)
        with cache._condition:
            cache._free_staging_slot_locked(0, ticket)

    def test_staging_wait_metric_accumulates_one_full_buffer_episode(self) -> None:
        cache = self._cache(1, staging_slots=1)
        self.addCleanup(cache.close)
        blocker = ExpertLoadTicket(
            key=ExpertKey(0, 0),
            queued_at=0.0,
            request_clock=0,
        )
        ticket = ExpertLoadTicket(
            key=ExpertKey(0, 1),
            queued_at=0.0,
            request_clock=0,
        )
        with cache._condition:
            cache._stage_owner[0] = blocker
            cache._stage_state[0] = "reading"

        def release_staging(*_args: object, **_kwargs: object) -> bool:
            cache._stage_owner[0] = None
            cache._stage_state[0] = "free"
            return True

        with (
            patch(
                "moevm.paged_runtime.time.perf_counter",
                side_effect=(100.0, 106.25),
            ),
            patch.object(
                cache._condition,
                "wait",
                side_effect=release_staging,
            ),
        ):
            self.assertEqual(cache._claim_staging_slot(ticket), 0)

        metrics = cache.metrics()
        self.assertEqual(metrics.staging_waits, 1)
        self.assertEqual(metrics.staging_wait_seconds, 6.25)
        with cache._condition:
            cache._free_staging_slot_locked(0, ticket)

    def test_async_forward_releases_lease_when_next_submit_fails(self) -> None:
        hidden_states = torch.ones(2, self.hidden_size)
        top_k_index = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
        top_k_weights = torch.ones(2, 2)
        cache = self._cache(
            1,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        original_submit = cache._submit_lookahead
        submissions = 0

        def fail_third_submit(key):
            nonlocal submissions
            submissions += 1
            if submissions == 3:
                raise RuntimeError("injected submit failure")
            return original_submit(key)

        cache._submit_lookahead = fail_third_submit
        try:
            with self.assertRaisesRegex(RuntimeError, "injected submit failure"):
                runtime.forward(
                    0,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                )
        finally:
            cache._submit_lookahead = original_submit

        cache.wait_idle()
        self.assertEqual(cache._slot_pin_count, [0])
        self.assertEqual(cache._pending_by_key, {})

    def test_async_stale_resident_ticket_reloads_without_pinning_deadlock(self) -> None:
        cache = self._cache(
            1,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        cache.get(one)

        missing = cache.submit(zero)
        future_hit = cache.submit(one)
        cache.resolve(missing)
        loaded = cache.resolve(future_hit)

        gate, up, down = self.original[one]
        torch.testing.assert_close(loaded.gate_up, torch.cat((gate, up)))
        torch.testing.assert_close(loaded.down, down)
        self.assertEqual(cache.resident_keys, (one,))
        metrics = cache.metrics()
        self.assertEqual(metrics.requests, 3)
        self.assertEqual(metrics.hits, 0)
        self.assertEqual(metrics.misses, 3)
        self.assertEqual(metrics.storage_bytes, 3 * self.store.spec.size_bytes)
        self.assertEqual(metrics.storage_loads, 3)

    def test_discarded_lookahead_requeue_reserves_timeline_before_enqueue(
        self,
    ) -> None:
        class FakeEvent:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def record(self, _stream: object) -> None:
                return None

        cache = self._cache(1, staging_slots=1, pipeline_mode="async")
        self.addCleanup(cache.close)
        key = ExpertKey(0, 0)
        ticket = ExpertLoadTicket(
            key=key,
            queued_at=time.perf_counter(),
            request_clock=1,
            state="discarded",
            is_lookahead=True,
            demanded=True,
        )
        observed_enqueue = threading.Event()
        original_enqueue = cache._enqueue_ticket

        with patch("moevm.paged_runtime.torch.cuda.Event", FakeEvent):
            capture = _CudaTimelineCapture(torch.device("cuda"))
            capture.begin(object())

            def observe_enqueue(queued: ExpertLoadTicket, **_kwargs: object) -> None:
                self.assertIs(queued, ticket)
                self.assertIs(queued.cuda_timeline, capture)
                ledger_entry = capture._transfer_ledger.get(id(queued))
                self.assertIsNotNone(ledger_entry)
                self.assertIs(ledger_entry.ticket, queued)
                observed_enqueue.set()

            cache._enqueue_ticket = observe_enqueue  # type: ignore[method-assign]
            try:
                requeued = cache._requeue_discarded_lookahead(
                    ticket,
                    cuda_timeline=capture,
                )
            finally:
                cache._enqueue_ticket = original_enqueue  # type: ignore[method-assign]
                with cache._condition:
                    cache._pending_by_key.pop(key, None)
                    ticket.state = "discarded"
                capture.cancel_transfer(ticket, "test cleanup")

        self.assertIs(requeued, ticket)
        self.assertTrue(observed_enqueue.is_set())

    def test_async_stale_hit_promotes_a_concurrent_lookahead_to_demand(self) -> None:
        cache = self._cache(
            1,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        cache.get(zero)
        stale_hit = cache.submit(zero)
        self.assertTrue(stale_hit.counted_as_hit)
        cache.get(one)

        lookahead = cache._submit_lookahead(zero)
        self.assertTrue(lookahead.storage_done.wait(timeout=2.0))
        self.assertFalse(lookahead.demanded)
        self.assertIs(cache._pending_by_key[zero], lookahead)

        with cache.execution_lock:
            refreshed = cache._refresh_stale_resident_ticket(stale_hit)
            self.assertIs(refreshed, lookahead)
            self.assertTrue(lookahead.demanded)
            self.assertEqual(lookahead.request_clock, stale_hit.request_clock)
            self.assertFalse(lookahead.discard_requested)
            lease = cache._acquire_ticket(
                refreshed,
                compute_stream=None,
                synchronize=True,
            )
            lease.release_after()

        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (zero,))

    def test_async_stale_ticket_cannot_requeue_after_close(self) -> None:
        cache = self._cache(1, pipeline_mode="async")
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        cache.get(zero)
        stale = cache.submit(zero)
        cache.get(one)
        cache.close()

        with self.assertRaisesRegex(RuntimeError, "expert slot cache is closed"):
            cache.resolve(stale)
        self.assertEqual(cache._pending_by_key, {})

    def test_async_storage_failure_preserves_previous_victim(self) -> None:
        cache = self._cache(1, pipeline_mode="async")
        self.addCleanup(cache.close)
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        cache.get(zero)
        original_load_into = self.store.load_into

        def fail_read(*_args, **_kwargs):
            raise OSError("injected async read failure")

        self.store.load_into = fail_read
        try:
            with self.assertRaisesRegex(OSError, "injected async read failure"):
                cache.get(one)
        finally:
            self.store.load_into = original_load_into

        self.assertEqual(cache.resident_keys, (zero,))
        cache.get(zero)
        metrics = cache.metrics()
        self.assertEqual(metrics.storage_failures, 1)
        self.assertEqual(metrics.hits, 1)

    def test_async_wait_idle_reports_unobserved_storage_failure(self) -> None:
        cache = self._cache(1, pipeline_mode="async")
        self.addCleanup(cache.close)
        original_load_into = self.store.load_into

        def fail_read(*_args, **_kwargs):
            raise OSError("unobserved async read failure")

        self.store.load_into = fail_read
        try:
            ticket = cache.submit(ExpertKey(0, 0))
            self.assertTrue(ticket.completed.wait(timeout=2))
        finally:
            self.store.load_into = original_load_into

        with self.assertRaisesRegex(OSError, "unobserved async read failure"):
            cache.wait_idle()
        cache.close()

    def test_async_transfer_failure_leaves_slot_empty_and_recovers(self) -> None:
        cache = self._cache(1, pipeline_mode="async")
        self.addCleanup(cache.close)
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        cache.get(zero)
        original_copy = cache._copy_staging_to_slot

        def fail_copy(*_args, **_kwargs):
            raise RuntimeError("injected async transfer failure")

        cache._copy_staging_to_slot = fail_copy
        try:
            with self.assertRaisesRegex(
                RuntimeError, "injected async transfer failure"
            ):
                cache.get(one)
        finally:
            cache._copy_staging_to_slot = original_copy

        self.assertEqual(cache.resident_keys, ())
        cache.get(one)
        self.assertEqual(cache.resident_keys, (one,))
        metrics = cache.metrics()
        self.assertEqual(metrics.evictions, 1)
        self.assertEqual(metrics.transfer_failures, 1)

    def test_async_close_waits_for_inflight_storage(self) -> None:
        cache = self._cache(1, pipeline_mode="async")
        key = ExpertKey(0, 0)
        original_load_into = self.store.load_into
        entered = threading.Event()
        release = threading.Event()
        close_errors: list[Exception] = []

        def blocked_load(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release storage worker")
            return original_load_into(*args, **kwargs)

        self.store.load_into = blocked_load
        try:
            cache.submit(key)
            self.assertTrue(entered.wait(timeout=2))

            def close_cache() -> None:
                try:
                    cache.close()
                except Exception as exc:  # noqa: BLE001 - capture thread result
                    close_errors.append(exc)

            closer = threading.Thread(target=close_cache)
            closer.start()
            time.sleep(0.02)
            self.assertTrue(closer.is_alive())
            release.set()
            closer.join(timeout=5)
            self.assertFalse(closer.is_alive())
        finally:
            release.set()
            self.store.load_into = original_load_into
            cache.close()

        self.assertEqual(close_errors, [])
        self.assertEqual(cache.resident_keys, (key,))
        self.assertEqual(cache.metrics().storage_failures, 0)

    def test_async_queue_full_rejects_without_holding_execution_lock(self) -> None:
        cache = self._cache(
            1,
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        first = cache.submit(ExpertKey(0, 0))
        self.assertTrue(first.storage_done.wait(timeout=2))
        second = cache.submit(ExpertKey(0, 1))

        deadline = time.monotonic() + 2
        while cache._jobs.qsize() != 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(cache._jobs.qsize(), 0)
        third = cache.submit(ExpertKey(0, 2))

        started = time.perf_counter()
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            cache.submit(ExpertKey(0, 3))
        self.assertLess(time.perf_counter() - started, 0.5)

        cache.resolve(first)
        cache.wait_idle()
        self.assertTrue(second.completed.is_set())
        self.assertTrue(third.completed.is_set())
        metrics = cache.metrics()
        self.assertEqual(metrics.requests, 3)
        self.assertEqual(metrics.misses, 3)
        self.assertEqual(metrics.admission_rejections, 1)

    def test_static_policy_preserves_static_expert(self) -> None:
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        two = ExpertKey(0, 2)
        cache = self._cache(
            2,
            policy=CachePolicy.STATIC,
            static_keys=(zero,),
        )

        cache.get(zero)
        cache.get(one)
        cache.get(two)

        self.assertEqual(cache.resident_keys, (zero, two))
        self.assertEqual(cache.metrics().evictions, 1)

    def test_hybrid_policy_uses_lru_only_in_dynamic_partition(self) -> None:
        zero = ExpertKey(0, 0)
        one = ExpertKey(0, 1)
        two = ExpertKey(0, 2)
        three = ExpertKey(0, 3)
        cache = self._cache(
            3,
            policy=CachePolicy.HYBRID,
            static_keys=(zero,),
        )

        cache.get(zero)
        cache.get(one)
        cache.get(two)
        cache.get(one)
        cache.get(three)

        self.assertEqual(cache.resident_keys, (zero, one, three))
        self.assertEqual(cache.metrics().evictions, 1)

    def test_capacity_per_layer_prevents_cross_layer_eviction(self) -> None:
        cache = ExpertSlotCache(
            self.store,
            capacity_per_layer=2,
            device="cpu",
            policy=CachePolicy.LRU,
            pin_staging=False,
        )
        layer_zero = (ExpertKey(0, 0), ExpertKey(0, 1))
        layer_one = (ExpertKey(1, 0), ExpertKey(1, 1))
        cache.prefetch((*layer_zero, *layer_one))

        cache.get(layer_zero[0])
        cache.get(ExpertKey(0, 2))

        self.assertEqual(
            cache.resident_keys,
            (ExpertKey(0, 0), ExpertKey(0, 2), *layer_one),
        )
        self.assertEqual(cache.capacity_per_layer, 2)
        self.assertEqual(cache.capacity, 4)
        self.assertEqual(
            cache.allocated_cache_bytes,
            4 * self.store.spec.size_bytes,
        )

    def test_per_layer_hybrid_protects_static_and_isolates_dynamic_slots(self) -> None:
        static_keys = (ExpertKey(0, 0), ExpertKey(1, 0))
        cache = ExpertSlotCache(
            self.store,
            capacity_per_layer=2,
            device="cpu",
            policy=CachePolicy.HYBRID,
            static_keys=static_keys,
            pin_staging=False,
        )
        cache.prefetch((*static_keys, ExpertKey(0, 1), ExpertKey(1, 1)))

        cache.get(ExpertKey(0, 2))

        self.assertEqual(
            cache.resident_keys,
            (
                ExpertKey(0, 0),
                ExpertKey(0, 2),
                ExpertKey(1, 0),
                ExpertKey(1, 1),
            ),
        )
        self.assertEqual(cache.metrics().evictions, 1)

    def test_cpu_paged_forward_matches_full_expert_reference(self) -> None:
        generator = torch.Generator().manual_seed(7)
        hidden_states = torch.randn(5, self.hidden_size, generator=generator)
        top_k_index = torch.tensor(
            [[0, 1], [2, 1], [3, 0], [1, 3], [2, 0]], dtype=torch.long
        )
        top_k_weights = torch.rand(5, 2, generator=generator)
        runtime = PagedExpertRuntime(self._cache(2))

        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        actual = runtime.forward(
            0,
            hidden_states,
            top_k_index,
            top_k_weights,
        )

        torch.testing.assert_close(actual, expected)
        metrics = runtime.metrics()
        self.assertEqual(metrics.requests, 4)
        self.assertEqual(metrics.misses, 4)
        self.assertEqual(metrics.evictions, 2)
        self.assertGreater(metrics.forward_seconds, 0.0)

    def test_forward_rejects_unknown_layer_before_sentinel_handling(self) -> None:
        runtime = PagedExpertRuntime(self._cache(1))
        hidden_states = torch.zeros(1, self.hidden_size)
        top_k_index = torch.tensor([[0]])
        top_k_weights = torch.ones(1, 1)

        with self.assertRaisesRegex(KeyError, "unknown expert layer"):
            runtime.forward(99, hidden_states, top_k_index, top_k_weights)

    def test_transformers_forward_adapter_uses_attached_runtime(self) -> None:
        runtime = PagedExpertRuntime(self._cache(2))

        class ExpertModule:
            _moevm_paged_runtime = runtime
            _moevm_layer_index = 0

        hidden_states = torch.zeros(1, self.hidden_size)
        top_k_index = torch.tensor([[0]])
        top_k_weights = torch.ones(1, 1)

        register_transformers_paged_experts("moevm_paged_test")
        actual = transformers_paged_experts_forward(
            ExpertModule(), hidden_states, top_k_index, top_k_weights
        )
        expected = self._reference_forward(hidden_states, top_k_index, top_k_weights)
        torch.testing.assert_close(actual, expected)

    def test_transformers_adapter_rejects_unknown_model_type(self) -> None:
        class Config:
            model_type = "unsupported_moe"

        class Model:
            config = Config()

        with self.assertRaisesRegex(
            TypeError,
            "unsupported Transformers MoE model type.*mixtral, olmoe, qwen2_moe",
        ):
            transformers_moe_adapter_for_model(Model())

    def test_transformers_adapter_rejects_boolean_expert_width(self) -> None:
        class Config:
            moe_intermediate_size = True

        adapter = TransformersMoEAdapter(
            "qwen2_moe",
            "Qwen2MoE",
            expert_intermediate_size_attr="moe_intermediate_size",
        )
        with self.assertRaisesRegex(TypeError, "no positive moe_intermediate_size"):
            adapter.expert_intermediate_size(Config())

    def test_runtime_attaches_to_a_meta_initialized_olmoe(self) -> None:
        try:
            from accelerate import init_empty_weights
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError:
            self.skipTest("requires accelerate and transformers")
        config = OlmoeConfig(
            vocab_size=16,
            eos_token_id=2,
            pad_token_id=1,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
        )
        with init_empty_weights(include_buffers=True):
            model = OlmoeForCausalLM(config)
        runtime = PagedExpertRuntime(
            ExpertSlotCache(
                self.store,
                capacity_per_layer=1,
                device="meta",
                pin_staging=False,
            )
        )

        attach_transformers_olmoe_runtime(
            model,
            runtime,
            implementation="moevm_paged_meta_test",
        )

        experts = model.model.layers[0].mlp.experts
        self.assertIs(experts._moevm_paged_runtime, runtime)
        self.assertEqual(experts._moevm_layer_index, 0)
        self.assertEqual(config._experts_implementation, "moevm_paged_meta_test")

        config.hidden_act = "gelu"
        with self.assertRaisesRegex(ValueError, "requires SiLU"):
            attach_transformers_olmoe_runtime(
                model,
                runtime,
                implementation="moevm_paged_invalid_activation_test",
            )

    def test_tiny_mixtral_meta_loader_forward_matches_eager(self) -> None:
        try:
            from accelerate import init_empty_weights
            from transformers import MixtralConfig, MixtralForCausalLM
        except ImportError:
            self.skipTest("requires accelerate and transformers")

        torch.manual_seed(271828)
        config = MixtralConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            eos_token_id=None,
            pad_token_id=1,
        )
        eager_model = MixtralForCausalLM(config).eval()
        state = eager_model.state_dict()
        checkpoint: dict[str, torch.Tensor] = {}
        expert_parameter_names: set[str] = set()
        for layer in range(config.num_hidden_layers):
            gate_up_name = f"model.layers.{layer}.mlp.experts.gate_up_proj"
            down_name = f"model.layers.{layer}.mlp.experts.down_proj"
            expert_parameter_names.update((gate_up_name, down_name))
            gate_up = state[gate_up_name]
            down = state[down_name]
            for expert in range(config.num_local_experts):
                gate, up = gate_up[expert].chunk(2, dim=0)
                prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                checkpoint[f"{prefix}.gate_proj.weight"] = gate.contiguous()
                checkpoint[f"{prefix}.up_proj.weight"] = up.contiguous()
                checkpoint[f"{prefix}.down_proj.weight"] = down[expert].contiguous()
        for tensor_name, tensor in state.items():
            if tensor_name not in expert_parameter_names:
                checkpoint[tensor_name] = tensor.contiguous()

        snapshot = self.snapshot / "tiny-mixtral"
        snapshot.mkdir()
        shard_name = "model.safetensors"
        save_file(checkpoint, snapshot / shard_name)
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        tensor_name: shard_name for tensor_name in checkpoint
                    },
                }
            ),
            encoding="utf-8",
        )

        paged_config = MixtralConfig.from_dict(config.to_dict())
        with init_empty_weights(include_buffers=False):
            paged_model = MixtralForCausalLM(paged_config)
        with SafetensorExpertStore(snapshot) as store:
            runtime = PagedExpertRuntime(
                ExpertSlotCache(
                    store,
                    capacity_per_layer=config.num_local_experts,
                    device="cpu",
                    policy=CachePolicy.LRU,
                    pin_staging=False,
                )
            )
            adapter = attach_transformers_moe_runtime(
                paged_model,
                runtime,
                implementation="moevm_paged_tiny_mixtral",
            )
            self.assertEqual(adapter.model_type, "mixtral")
            self.assertEqual(
                transformers_moe_adapter_for_model(paged_model),
                adapter,
            )
            loaded = load_non_expert_weights_into_meta_model(
                paged_model,
                store,
                device="cpu",
            )
            paged_model.eval()

            self.assertEqual(set(loaded), set(state) - expert_parameter_names)
            validation = validate_transformers_paged_model(
                paged_model,
                store,
                runtime=runtime,
            )
            self.assertEqual(validation["adapter"], "mixtral")
            self.assertEqual(validation["expert_meta_parameters"], 4)

            input_ids = torch.tensor([[2, 7, 11, 5]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                eager_logits = eager_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                paged_logits = paged_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            torch.testing.assert_close(paged_logits, eager_logits)

    def test_tiny_qwen2_moe_shared_expert_forward_matches_eager(self) -> None:
        try:
            from accelerate import init_empty_weights
            from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
        except ImportError:
            self.skipTest("requires accelerate and transformers")

        torch.manual_seed(161803)
        config = Qwen2MoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            moe_intermediate_size=4,
            shared_expert_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            eos_token_id=None,
            pad_token_id=1,
        )
        eager_model = Qwen2MoeForCausalLM(config).eval()
        state = eager_model.state_dict()
        checkpoint: dict[str, torch.Tensor] = {}
        expert_parameter_names: set[str] = set()
        for layer in range(config.num_hidden_layers):
            gate_up_name = f"model.layers.{layer}.mlp.experts.gate_up_proj"
            down_name = f"model.layers.{layer}.mlp.experts.down_proj"
            expert_parameter_names.update((gate_up_name, down_name))
            gate_up = state[gate_up_name]
            down = state[down_name]
            for expert in range(config.num_experts):
                gate, up = gate_up[expert].chunk(2, dim=0)
                prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                checkpoint[f"{prefix}.gate_proj.weight"] = gate.contiguous()
                checkpoint[f"{prefix}.up_proj.weight"] = up.contiguous()
                checkpoint[f"{prefix}.down_proj.weight"] = down[expert].contiguous()
        for tensor_name, tensor in state.items():
            if tensor_name not in expert_parameter_names:
                checkpoint[tensor_name] = tensor.contiguous()

        snapshot = self.snapshot / "tiny-qwen2-moe"
        snapshot.mkdir()
        shard_name = "model.safetensors"
        save_file(checkpoint, snapshot / shard_name)
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        tensor_name: shard_name for tensor_name in checkpoint
                    },
                }
            ),
            encoding="utf-8",
        )

        paged_config = Qwen2MoeConfig.from_dict(config.to_dict())
        with init_empty_weights(include_buffers=False):
            paged_model = Qwen2MoeForCausalLM(paged_config)
        with SafetensorExpertStore(snapshot) as store:
            runtime = PagedExpertRuntime(
                ExpertSlotCache(
                    store,
                    capacity_per_layer=config.num_experts,
                    device="cpu",
                    policy=CachePolicy.LRU,
                    pin_staging=False,
                )
            )
            with self.assertRaisesRegex(
                TypeError,
                "expected an OLMoEForCausalLM-compatible model",
            ):
                attach_transformers_olmoe_runtime(paged_model, runtime)
            paged_model.config.moe_intermediate_size += 1
            with self.assertRaisesRegex(
                ValueError,
                "model dimensions do not match the expert store",
            ):
                attach_transformers_moe_runtime(
                    paged_model,
                    runtime,
                    implementation="moevm_paged_tiny_qwen2_moe",
                )
            paged_model.config.moe_intermediate_size -= 1
            adapter = attach_transformers_moe_runtime(
                paged_model,
                runtime,
                implementation="moevm_paged_tiny_qwen2_moe",
            )
            self.assertEqual(adapter.model_type, "qwen2_moe")
            loaded = load_non_expert_weights_into_meta_model(
                paged_model,
                store,
                device="cpu",
            )
            paged_model.eval()

            self.assertEqual(set(loaded), set(state) - expert_parameter_names)
            validation = validate_transformers_paged_model(
                paged_model,
                store,
                runtime=runtime,
            )
            self.assertEqual(validation["adapter"], "qwen2_moe")
            self.assertEqual(validation["expert_meta_parameters"], 4)
            self.assertTrue(
                any("shared_expert" in tensor_name for tensor_name in loaded)
            )

            input_ids = torch.tensor([[2, 7, 11, 5]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                eager_logits = eager_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                paged_logits = paged_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            torch.testing.assert_close(
                paged_logits,
                eager_logits,
                rtol=0.0,
                atol=0.0,
            )

    def test_tiny_olmoe_meta_loader_forward_and_generate_match_eager(self) -> None:
        try:
            from accelerate import init_empty_weights
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError:
            self.skipTest("requires accelerate and transformers")

        torch.manual_seed(314159)
        config = OlmoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            eos_token_id=None,
            pad_token_id=1,
        )
        eager_model = OlmoeForCausalLM(config).eval()
        state = eager_model.state_dict()
        checkpoint: dict[str, torch.Tensor] = {}
        expert_parameter_names: set[str] = set()
        for layer in range(config.num_hidden_layers):
            gate_up_name = f"model.layers.{layer}.mlp.experts.gate_up_proj"
            down_name = f"model.layers.{layer}.mlp.experts.down_proj"
            expert_parameter_names.update((gate_up_name, down_name))
            gate_up = state[gate_up_name]
            down = state[down_name]
            for expert in range(config.num_experts):
                gate, up = gate_up[expert].chunk(2, dim=0)
                prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                checkpoint[f"{prefix}.gate_proj.weight"] = gate.contiguous()
                checkpoint[f"{prefix}.up_proj.weight"] = up.contiguous()
                checkpoint[f"{prefix}.down_proj.weight"] = down[expert].contiguous()
        for tensor_name, tensor in state.items():
            if tensor_name not in expert_parameter_names:
                checkpoint[tensor_name] = tensor.contiguous()

        tiny_snapshot = self.snapshot / "tiny-olmoe"
        tiny_snapshot.mkdir()
        shard_name = "model.safetensors"
        save_file(checkpoint, tiny_snapshot / shard_name)
        (tiny_snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        tensor_name: shard_name for tensor_name in checkpoint
                    },
                }
            ),
            encoding="utf-8",
        )

        paged_config = OlmoeConfig.from_dict(config.to_dict())
        with init_empty_weights(include_buffers=False):
            paged_model = OlmoeForCausalLM(paged_config)
        with SafetensorExpertStore(tiny_snapshot) as tiny_store:
            runtime = PagedExpertRuntime(
                ExpertSlotCache(
                    tiny_store,
                    capacity_per_layer=config.num_experts,
                    device="cpu",
                    policy=CachePolicy.LRU,
                    pin_staging=False,
                )
            )
            attach_transformers_olmoe_runtime(
                paged_model,
                runtime,
                implementation="moevm_paged_tiny_e2e",
            )
            loaded = load_non_expert_weights_into_meta_model(
                paged_model,
                tiny_store,
                device="cpu",
            )
            paged_model.eval()

            remaining_meta = {
                name
                for name, parameter in paged_model.named_parameters()
                if parameter.device.type == "meta"
            }
            self.assertEqual(remaining_meta, expert_parameter_names)
            self.assertEqual(set(loaded), set(state) - expert_parameter_names)
            validation = validate_transformers_paged_model(paged_model, tiny_store)
            self.assertEqual(validation["expert_meta_parameters"], 4)
            self.assertEqual(validation["dtype"], "torch.float32")

            input_ids = torch.tensor([[2, 7, 11, 5]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                eager_logits = eager_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                paged_logits = paged_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            torch.testing.assert_close(paged_logits, eager_logits)

            eager_tokens = eager_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                do_sample=False,
                use_cache=False,
            )
            paged_tokens = paged_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                do_sample=False,
                use_cache=False,
            )
            torch.testing.assert_close(paged_tokens, eager_tokens)
            self.assertGreater(runtime.metrics().requests, 0)

    def test_non_expert_loader_materializes_only_requested_meta_parameter(self) -> None:
        try:
            import accelerate  # noqa: F401
        except ImportError:
            self.skipTest("requires accelerate")

        class MetaHolder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.other = torch.nn.Linear(2, 2, bias=False, device="meta")

        model = MetaHolder()
        loaded = load_non_expert_weights_into_meta_model(
            model,
            self.store,
            device="cpu",
        )

        self.assertEqual(loaded, ("other.weight",))
        self.assertEqual(model.other.weight.device.type, "cpu")
        self.assertEqual(model.other.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(model.other.weight, self.non_expert)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_small_cuda_forward_matches_reference(self) -> None:
        device = "cuda"
        generator = torch.Generator().manual_seed(91)
        hidden_states = torch.randn(3, self.hidden_size, generator=generator).to(device)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]], device=device)
        top_k_weights = torch.rand(3, 2, generator=generator).to(device)
        runtime = PagedExpertRuntime(self._cache(2, device=device))

        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        actual = runtime.forward(
            0,
            hidden_states,
            top_k_index,
            top_k_weights,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        metrics = runtime.metrics()
        self.assertEqual(metrics.misses, 3)
        self.assertEqual(metrics.hits, 0)
        self.assertEqual(metrics.evictions, 1)
        self.assertEqual(
            metrics.host_to_device_bytes,
            3 * self.store.spec.size_bytes,
        )

        hit_output = runtime.forward(
            0,
            hidden_states[:1],
            torch.tensor([[2]], device=device),
            torch.ones(1, 1, device=device),
        )
        hit_expected = self._reference_forward(
            hidden_states[:1],
            torch.tensor([[2]], device=device),
            torch.ones(1, 1, device=device),
        )
        torch.testing.assert_close(hit_output, hit_expected, rtol=1e-5, atol=1e-6)
        hit_metrics = runtime.metrics()
        self.assertEqual(hit_metrics.hits, 1)
        self.assertEqual(hit_metrics.misses, 3)
        self.assertEqual(hit_metrics.evictions, 1)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_forward_uses_events_and_reuses_staging(self) -> None:
        device = "cuda"
        generator = torch.Generator().manual_seed(919)
        hidden_states = torch.randn(3, self.hidden_size, generator=generator).to(device)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]], device=device)
        top_k_weights = torch.rand(3, 2, generator=generator).to(device)
        cache = self._cache(
            1,
            device=device,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)

        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        torch.cuda.current_stream().synchronize()
        original_synchronize = torch.cuda.synchronize

        def reject_global_synchronize(*_args, **_kwargs):
            raise AssertionError("async forward used a global CUDA synchronize")

        torch.cuda.synchronize = reject_global_synchronize
        try:
            actual = runtime.forward(
                0,
                hidden_states,
                top_k_index,
                top_k_weights,
            )
        finally:
            torch.cuda.synchronize = original_synchronize

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        cache.wait_idle()
        metrics = cache.metrics()
        self.assertEqual(metrics.misses, 3)
        self.assertEqual(metrics.evictions, 2)
        self.assertEqual(
            metrics.host_to_device_bytes,
            3 * self.store.spec.size_bytes,
        )
        self.assertLessEqual(metrics.peak_staging_in_use, 2)
        self.assertEqual(cache._stage_state, ["free", "free"])
        self.assertTrue(any(event is not None for event in cache._slot_last_use_event))
        with self.assertRaisesRegex(RuntimeError, "raw CUDA weights are unsafe"):
            cache.get(ExpertKey(0, 0))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_worker_h2d_reserves_only_until_demand(self) -> None:
        device = "cuda"
        cache = self._cache(
            2,
            device=device,
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        key = ExpertKey(0, 0)

        ticket = cache._submit_lookahead(key)
        self.assertTrue(ticket.transfer_enqueued.wait(timeout=2.0))
        self.assertIsNone(ticket.error)
        with cache._condition:
            slot = ticket.destination_slot
            self.assertIsNotNone(slot)
            self.assertFalse(ticket.demanded)
            self.assertTrue(ticket.reservation_active)
            self.assertIs(cache._slot_reservation[slot], ticket)
            self.assertIsNone(cache._slot_to_key[slot])
            self.assertNotIn(key, cache._key_to_slot)
            self.assertEqual(cache.metrics().evictions, 0)

        torch.cuda.synchronize(device)
        cache._poll_transfer_completions()
        with cache._condition:
            self.assertEqual(ticket.state, "prefetched")
            self.assertTrue(ticket.reservation_active)
            self.assertIs(cache._slot_reservation[slot], ticket)
            self.assertEqual(cache._stage_state, ["free"])
            self.assertIsNone(cache._slot_to_key[slot])
            self.assertNotIn(key, cache._key_to_slot)

        stream = torch.cuda.current_stream(device)
        with cache.execution_lock:
            lease = cache._acquire_ticket(
                ticket,
                compute_stream=stream,
                synchronize=False,
                account_demand=True,
            )
            self.assertEqual(cache._key_to_slot[key], slot)
            self.assertEqual(cache._slot_to_key[slot], key)
            stream.synchronize()
            cache._poll_transfer_completions()
            lease.release_after(stream)

        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (key,))
        self.assertEqual(cache._slot_reservation, [None, None])
        self.assertEqual(cache._stage_state, ["free"])
        metrics = cache.metrics()
        self.assertEqual((metrics.requests, metrics.hits, metrics.misses), (1, 0, 1))
        self.assertEqual(metrics.evictions, 0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_counts_proactive_h2d_decline_without_empty_slot(self) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        resident = ExpertKey(0, 0)
        lookahead = ExpertKey(0, 1)

        cache.set_pipeline_mode("sync")
        cache.get(resident)
        cache.set_pipeline_mode("async")
        ticket = cache._submit_lookahead(lookahead)
        self.assertTrue(ticket.storage_done.wait(timeout=2.0))

        self.assertIsNone(ticket.destination_slot)
        self.assertEqual(cache.metrics().proactive_h2d_slot_declines, 1)

        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (resident,))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_prefetched_slot_is_not_available_to_another_demand(
        self,
    ) -> None:
        cache = self._cache(
            2,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        speculative = cache._submit_lookahead(ExpertKey(0, 0))
        self.assertTrue(speculative.transfer_enqueued.wait(timeout=2.0))
        torch.cuda.synchronize("cuda")
        cache._poll_transfer_completions()
        reserved_slot = speculative.destination_slot
        self.assertIsNotNone(reserved_slot)
        self.assertEqual(speculative.state, "prefetched")

        with cache.execution_lock:
            demanded = cache.submit(ExpertKey(0, 1))
            lease = cache._acquire_ticket(
                demanded,
                compute_stream=None,
                synchronize=True,
            )
            self.assertNotEqual(demanded.destination_slot, reserved_slot)
            self.assertIs(cache._slot_reservation[reserved_slot], speculative)
            self.assertIsNone(cache._slot_to_key[reserved_slot])
            lease.release_after()

        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (ExpertKey(0, 1),))
        self.assertEqual(cache._slot_reservation, [None, None])
        self.assertEqual(cache.metrics().evictions, 0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_demand_reclaims_an_undemanded_prefetched_reservation(
        self,
    ) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        speculative = cache._submit_lookahead(ExpertKey(0, 0))
        self.assertTrue(speculative.transfer_enqueued.wait(timeout=2.0))
        torch.cuda.synchronize("cuda")
        cache._poll_transfer_completions()
        self.assertEqual(speculative.state, "prefetched")

        demanded_key = ExpertKey(0, 1)
        with cache.execution_lock:
            demanded = cache.submit(demanded_key)
            lease = cache._acquire_ticket(
                demanded,
                compute_stream=None,
                synchronize=True,
            )
            lease.release_after()

        cache.wait_idle()
        self.assertEqual(speculative.state, "discarded")
        self.assertEqual(cache.resident_keys, (demanded_key,))
        self.assertEqual(cache._slot_reservation, [None])
        self.assertEqual(cache.metrics().evictions, 0)

        with cache.execution_lock:
            reloaded = cache._acquire_ticket(
                speculative,
                compute_stream=None,
                synchronize=True,
                account_demand=True,
            )
            reloaded.release_after()

        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (ExpertKey(0, 0),))
        self.assertEqual((cache.metrics().requests, cache.metrics().misses), (2, 2))
        self.assertEqual(cache.metrics().evictions, 1)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_speculative_h2d_failure_releases_reservation(self) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        original_enqueue = cache._enqueue_staging_to_slot

        def fail_enqueue(*_args, **_kwargs):
            raise RuntimeError("injected speculative H2D failure")

        cache._enqueue_staging_to_slot = fail_enqueue
        try:
            ticket = cache._submit_lookahead(ExpertKey(0, 0))
            self.assertTrue(ticket.completed.wait(timeout=2.0))
            with self.assertRaisesRegex(RuntimeError, "injected speculative H2D"):
                cache.wait_idle()
        finally:
            cache._enqueue_staging_to_slot = original_enqueue

        self.assertEqual(cache.resident_keys, ())
        self.assertEqual(cache._pending_by_key, {})
        self.assertEqual(cache._slot_reservation, [None])
        self.assertEqual(cache._slot_to_key, [None])
        self.assertEqual(cache._stage_state, ["free"])
        self.assertEqual(cache.metrics().transfer_failures, 1)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_poll_device_failure_releases_inflight_ticket(self) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        ticket = cache._submit_lookahead(ExpertKey(0, 0))
        self.assertTrue(ticket.transfer_enqueued.wait(timeout=2.0))
        original_set_device = torch.cuda.set_device

        def fail_set_device(*_args, **_kwargs):
            raise RuntimeError("injected CUDA device-selection failure")

        torch.cuda.set_device = fail_set_device
        try:
            cache._poll_transfer_completions()
        finally:
            torch.cuda.set_device = original_set_device

        self.assertTrue(ticket.completed.is_set())
        self.assertEqual(ticket.state, "failed")
        self.assertIsNotNone(ticket.error)
        self.assertEqual(cache._pending_by_key, {})
        self.assertEqual(cache._slot_reservation, [None])
        self.assertEqual(cache._slot_to_key, [None])
        self.assertEqual(cache._stage_state, ["free"])
        with self.assertRaisesRegex(RuntimeError, "device-selection failure"):
            cache.wait_idle()

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_poll_failure_quarantines_a_pinned_lease(self) -> None:
        """A polling failure must not recycle weights still queued for compute."""

        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        key = ExpertKey(0, 0)
        next_key = ExpertKey(0, 1)
        ticket = cache.submit(key)
        self.assertTrue(ticket.storage_done.wait(timeout=2.0))
        stream = torch.cuda.Stream(device="cuda")
        with cache.execution_lock:
            lease = cache._acquire_ticket(
                ticket,
                compute_stream=stream,
                synchronize=False,
            )
        with torch.cuda.stream(stream):
            # Queue work after the lease's ready-event wait.  The release
            # event below therefore represents a non-default compute tail.
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(100_000_000)
            else:  # pragma: no cover - CUDA builds normally expose _sleep
                torch.empty((1024, 1024), device="cuda").fill_(1)
        slot = ticket.destination_slot
        self.assertIsNotNone(slot)
        self.assertEqual(cache._slot_pin_count[slot], 1)

        original_set_device = torch.cuda.set_device

        def fail_set_device(*_args, **_kwargs):
            raise RuntimeError("injected CUDA device-selection failure")

        torch.cuda.set_device = fail_set_device
        try:
            cache._poll_transfer_completions()
        finally:
            torch.cuda.set_device = original_set_device

        with cache._condition:
            self.assertEqual(ticket.state, "failed")
            self.assertEqual(cache._slot_pin_count[slot], 1)
            self.assertTrue(cache._slot_quarantined[slot])
            self.assertIsNone(cache._slot_to_key[slot])
            self.assertNotIn(key, cache._key_to_slot)
            # Both allocators must reject the slot until the outstanding
            # compute lease records its tail event and releases it.
            self.assertIsNone(cache._select_async_slot_locked(next_key))
            self.assertIsNone(cache._select_proactive_slot_locked(next_key))

        with cache.execution_lock:
            lease.release_after(stream)

        with cache._condition:
            self.assertEqual(cache._slot_pin_count[slot], 0)
            self.assertFalse(cache._slot_quarantined[slot])
            self.assertEqual(cache._select_async_slot_locked(next_key), slot)

        tail_event = cache._slot_last_use_event[slot]
        self.assertIsNotNone(tail_event)
        # A real CUDA device-selection failure is terminal.  Reset only the
        # test fixture's error latch to exercise the allocator's ordering
        # guarantee after recovery; production never treats this as success.
        with cache._condition:
            cache._pipeline_error = None
            cache._unobserved_errors.clear()

        next_ticket = cache.submit(next_key)
        self.assertTrue(next_ticket.storage_done.wait(timeout=2.0))
        observed_wait_events: list[object] = []
        original_enqueue = cache._enqueue_staging_to_slot

        def capture_wait_events(*args, **kwargs):
            wait_events = kwargs.get("wait_events")
            if wait_events is None:
                wait_events = args[3]
            observed_wait_events.extend(wait_events)
            return original_enqueue(*args, **kwargs)

        cache._enqueue_staging_to_slot = capture_wait_events
        try:
            with cache.execution_lock:
                next_lease = cache._acquire_ticket(
                    next_ticket,
                    compute_stream=torch.cuda.current_stream("cuda"),
                    synchronize=False,
                )
                next_lease.release_after(torch.cuda.current_stream("cuda"))
        finally:
            cache._enqueue_staging_to_slot = original_enqueue

        self.assertEqual(next_ticket.destination_slot, slot)
        self.assertIn(tail_event, observed_wait_events)
        cache.wait_idle()

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_sync_cuda_empty_slot_waits_for_failed_lease_tail(self) -> None:
        """A sync-mode copy waits even when failure removed the logical key."""

        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        key = ExpertKey(0, 0)
        next_key = ExpertKey(0, 1)
        ticket = cache.submit(key)
        self.assertTrue(ticket.storage_done.wait(timeout=2.0))
        stream = torch.cuda.Stream(device="cuda")
        with cache.execution_lock:
            lease = cache._acquire_ticket(
                ticket,
                compute_stream=stream,
                synchronize=False,
            )
        with torch.cuda.stream(stream):
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(100_000_000)
            else:  # pragma: no cover - CUDA builds normally expose _sleep
                torch.empty((1024, 1024), device="cuda").fill_(1)
        slot = ticket.destination_slot
        self.assertIsNotNone(slot)
        lease.release_after(stream)
        tail_event = cache._slot_last_use_event[slot]
        self.assertIsNotNone(tail_event)

        original_set_device = torch.cuda.set_device

        def fail_set_device(*_args, **_kwargs):
            raise RuntimeError("injected CUDA device-selection failure")

        torch.cuda.set_device = fail_set_device
        try:
            cache._poll_transfer_completions()
        finally:
            torch.cuda.set_device = original_set_device

        with cache._condition:
            self.assertEqual(ticket.state, "failed")
            self.assertFalse(cache._slot_quarantined[slot])
            self.assertIs(cache._slot_last_use_event[slot], tail_event)
            cache._pipeline_error = None
            cache._unobserved_errors.clear()

        # This mirrors a recovered/drained mode switch.  `get()` must perform
        # its pre-copy synchronization even though the failed slot is empty in
        # the logical maps.
        cache.set_pipeline_mode("sync")
        synchronized_before_copy = False
        original_synchronize = cache._synchronize_device
        original_copy = cache._copy_staging_to_slot

        def track_synchronize() -> None:
            nonlocal synchronized_before_copy
            synchronized_before_copy = True
            original_synchronize()

        def assert_safe_copy(*args, **kwargs):
            self.assertTrue(synchronized_before_copy)
            self.assertTrue(tail_event.query())
            return original_copy(*args, **kwargs)

        cache._synchronize_device = track_synchronize
        cache._copy_staging_to_slot = assert_safe_copy
        try:
            cache.get(next_key)
        finally:
            cache._synchronize_device = original_synchronize
            cache._copy_staging_to_slot = original_copy

        self.assertEqual(cache.resident_keys, (next_key,))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_poll_device_failure_drains_ready_demand_ticket(self) -> None:
        cache = self._cache(
            2,
            device="cuda",
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        inflight = cache._submit_lookahead(ExpertKey(0, 0))
        self.assertTrue(inflight.transfer_enqueued.wait(timeout=2.0))
        ready_demand = cache.submit(ExpertKey(0, 1))
        self.assertTrue(ready_demand.storage_done.wait(timeout=2.0))
        self.assertEqual(ready_demand.state, "ready")
        original_set_device = torch.cuda.set_device

        def fail_set_device(*_args, **_kwargs):
            raise RuntimeError("injected CUDA device-selection failure")

        torch.cuda.set_device = fail_set_device
        try:
            cache._poll_transfer_completions()
        finally:
            torch.cuda.set_device = original_set_device

        with self.assertRaisesRegex(RuntimeError, "device-selection failure"):
            cache.wait_idle()
        self.assertEqual(inflight.state, "failed")
        self.assertEqual(ready_demand.state, "failed")
        self.assertEqual(cache._pending_by_key, {})
        self.assertEqual(cache._slot_reservation, [None, None])
        self.assertEqual(cache._slot_to_key, [None, None])
        self.assertEqual(cache._stage_state, ["free", "free"])

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_demand_waits_for_worker_to_publish_h2d_event(self) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        original_enqueue = cache._enqueue_staging_to_slot
        enqueue_entered = threading.Event()
        allow_enqueue = threading.Event()
        self.addCleanup(allow_enqueue.set)
        errors: list[Exception] = []

        def delayed_enqueue(*args, **kwargs):
            enqueue_entered.set()
            if not allow_enqueue.wait(timeout=2.0):
                raise TimeoutError("test did not release speculative H2D enqueue")
            return original_enqueue(*args, **kwargs)

        cache._enqueue_staging_to_slot = delayed_enqueue
        try:
            ticket = cache._submit_lookahead(ExpertKey(0, 0))
            self.assertTrue(enqueue_entered.wait(timeout=2.0))

            def demand_ticket() -> None:
                try:
                    with cache.execution_lock:
                        lease = cache._acquire_ticket(
                            ticket,
                            compute_stream=None,
                            synchronize=True,
                            account_demand=True,
                        )
                        lease.release_after()
                except Exception as exc:  # noqa: BLE001 - asserted below
                    errors.append(exc)

            demander = threading.Thread(target=demand_ticket)
            demander.start()
            deadline = time.monotonic() + 2.0
            while not ticket.demanded and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(ticket.demanded)
            self.assertFalse(ticket.transfer_enqueued.is_set())
            self.assertTrue(demander.is_alive())

            allow_enqueue.set()
            demander.join(timeout=2.0)
            self.assertFalse(demander.is_alive())
        finally:
            allow_enqueue.set()
            cache._enqueue_staging_to_slot = original_enqueue

        self.assertEqual(errors, [])
        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (ExpertKey(0, 0),))
        self.assertEqual(cache._stage_state, ["free"])
        self.assertEqual(cache._slot_reservation, [None])

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_async_cuda_waiting_demand_observes_terminal_enqueue_failure(
        self,
    ) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        enqueue_entered = threading.Event()
        allow_failure = threading.Event()
        self.addCleanup(allow_failure.set)
        errors: list[Exception] = []

        def delayed_failure(*_args, **_kwargs):
            enqueue_entered.set()
            if not allow_failure.wait(timeout=2.0):
                raise TimeoutError("test did not release speculative H2D failure")
            raise RuntimeError("injected delayed speculative H2D failure")

        original_enqueue = cache._enqueue_staging_to_slot
        cache._enqueue_staging_to_slot = delayed_failure
        try:
            key = ExpertKey(0, 0)
            ticket = cache._submit_lookahead(key)
            self.assertTrue(enqueue_entered.wait(timeout=2.0))

            def demand_ticket() -> None:
                try:
                    with cache.execution_lock:
                        cache._acquire_ticket(
                            ticket,
                            compute_stream=None,
                            synchronize=True,
                            account_demand=True,
                        )
                except Exception as exc:  # noqa: BLE001 - asserted below
                    errors.append(exc)

            demander = threading.Thread(target=demand_ticket)
            demander.start()
            deadline = time.monotonic() + 2.0
            while not ticket.demanded and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(ticket.demanded)
            self.assertFalse(ticket.transfer_enqueued.is_set())

            allow_failure.set()
            demander.join(timeout=2.0)
            self.assertFalse(demander.is_alive())
        finally:
            allow_failure.set()
            cache._enqueue_staging_to_slot = original_enqueue

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("injected delayed speculative H2D", str(errors[0]))
        self.assertTrue(ticket.transfer_enqueued.is_set())
        self.assertIsNotNone(ticket.error)
        self.assertEqual(cache.resident_keys, ())
        self.assertEqual(cache._pending_by_key, {})
        self.assertEqual(cache._slot_reservation, [None])
        self.assertEqual(cache._slot_to_key, [None])
        self.assertEqual(cache._stage_state, ["free"])

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_cuda_timeline_capture_drains_prior_worker_work_before_origin(self) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        prior_ticket = cache._submit_lookahead(ExpertKey(0, 0))
        self.assertTrue(prior_ticket.storage_done.wait(timeout=2.0))
        self.assertTrue(prior_ticket.transfer_enqueued.wait(timeout=2.0))

        with runtime.cuda_timeline_capture() as capture:
            # The scope itself needs no forward: entering it must first drain
            # the prior worker ticket so its H2D cannot enter this origin.
            pass

        self.assertIsNotNone(capture.result)
        self.assertTrue(capture.result["complete"])
        self.assertEqual(capture.result["schema_version"], 2)
        self.assertEqual(
            capture.result["coverage"],
            {"cache_transfer_loads_delta": 0, "h2d_span_count": 0},
        )
        self.assertEqual(cache._pending_by_key, {})
        self.assertEqual(cache._stage_state, ["free"])

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_cuda_timeline_capture_supports_proactive_worker_h2d(self) -> None:
        cache = self._cache(
            1,
            device="cuda",
            staging_slots=1,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        worker_enqueued_h2d = threading.Event()
        original_enqueue = cache._enqueue_staging_to_slot

        def observe_enqueue(*args, **kwargs):
            if threading.current_thread().name == "moevm-expert-io":
                worker_enqueued_h2d.set()
            return original_enqueue(*args, **kwargs)

        cache._enqueue_staging_to_slot = observe_enqueue  # type: ignore[method-assign]
        try:
            with runtime.cuda_timeline_capture() as capture:
                ticket = cache._submit_lookahead(
                    ExpertKey(0, 0),
                    cuda_timeline=capture,
                )
                self.assertTrue(ticket.storage_done.wait(timeout=2.0))
                self.assertTrue(worker_enqueued_h2d.wait(timeout=2.0))
                self.assertTrue(ticket.transfer_enqueued.wait(timeout=2.0))
                # Capture instrumentation must not publish the speculative
                # slot before a real demand accounts for it.
                self.assertEqual(cache.resident_keys, ())
                self.assertTrue(ticket.reservation_active)
        finally:
            cache._enqueue_staging_to_slot = original_enqueue  # type: ignore[method-assign]

        self.assertIsNotNone(capture.result)
        timeline = capture.result
        self.assertTrue(timeline["complete"])
        self.assertEqual(timeline["schema_version"], 2)
        h2d_span_count = sum(span["lane"] == "h2d" for span in timeline["spans"])
        self.assertGreater(h2d_span_count, 0)
        self.assertEqual(
            timeline["coverage"],
            {
                "cache_transfer_loads_delta": h2d_span_count,
                "h2d_span_count": h2d_span_count,
            },
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_cuda_timeline_stale_resident_lookahead_reload_has_coverage(
        self,
    ) -> None:
        """An evicted resident lookahead must trace its later demand reload."""

        cache = self._cache(
            1,
            device="cuda",
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        one = ExpertKey(0, 1)

        # Create a logical resident before the scope.  The first lookahead
        # below is therefore a cache-hit placeholder rather than a transfer.
        cache.set_pipeline_mode("sync")
        cache.get(one)
        cache.set_pipeline_mode("async")

        generator = torch.Generator().manual_seed(20260816)
        hidden_states = torch.randn(2, self.hidden_size, generator=generator).to(
            cache.device
        )
        top_k_index = torch.tensor([[0], [1]], device=cache.device)
        top_k_weights = torch.ones(2, 1, device=cache.device)
        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        submitted: dict[ExpertKey, ExpertLoadTicket] = {}
        original_submit = cache._submit_lookahead

        def observe_submit(
            key: ExpertKey,
            *,
            cuda_timeline: _CudaTimelineCapture | None = None,
        ) -> ExpertLoadTicket:
            ticket = original_submit(key, cuda_timeline=cuda_timeline)
            submitted[key] = ticket
            if key == one:
                self.assertEqual(ticket.state, "resident")
                self.assertTrue(ticket.is_lookahead)
                # A hit has no H2D yet, so admission is deliberately deferred
                # until eviction makes it worker-visible work.
                self.assertIsNone(ticket.cuda_timeline)
            return ticket

        cache._submit_lookahead = observe_submit  # type: ignore[method-assign]
        try:
            with runtime.cuda_timeline_capture() as capture:
                actual = runtime.forward(
                    0,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                )
                torch.cuda.synchronize(cache.device)
        finally:
            cache._submit_lookahead = original_submit  # type: ignore[method-assign]

        self.assertIsNotNone(capture.result)
        timeline = capture.result
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(timeline["complete"])
        self.assertEqual(timeline["schema_version"], 2)
        h2d_span_count = sum(span["lane"] == "h2d" for span in timeline["spans"])
        self.assertEqual(h2d_span_count, 2)
        self.assertEqual(
            timeline["coverage"],
            {"cache_transfer_loads_delta": 2, "h2d_span_count": 2},
        )
        self.assertIs(submitted[one].cuda_timeline, capture)

        cache.wait_idle()
        self.assertEqual(cache.resident_keys, (one,))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_cuda_timeline_capture_records_h2d_and_expert_compute(self) -> None:
        device = "cuda"
        generator = torch.Generator().manual_seed(920)
        hidden_states = torch.randn(3, self.hidden_size, generator=generator).to(device)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]], device=device)
        top_k_weights = torch.rand(3, 2, generator=generator).to(device)
        cache = self._cache(
            2,
            device=device,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)

        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        alternate_stream = torch.cuda.Stream(device=device)
        alternate_stream.wait_stream(torch.cuda.current_stream(device))
        with runtime.cuda_timeline_capture() as capture:
            with torch.cuda.stream(alternate_stream):
                actual = runtime.forward(
                    0,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                )
            torch.cuda.current_stream(device).wait_stream(alternate_stream)
            torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertIsNotNone(capture.result)
        timeline = capture.result
        self.assertEqual(timeline["status"], "measured")
        self.assertEqual(timeline["schema_version"], 2)
        self.assertGreater(timeline["summary"]["transfer"]["interval_count"], 0)
        self.assertGreater(timeline["summary"]["compute"]["interval_count"], 0)
        self.assertGreaterEqual(timeline["summary"]["overlap"]["duration_ms"], 0.0)
        self.assertTrue(all(span["layer"] == 0 for span in timeline["spans"]))
        h2d_span_count = sum(span["lane"] == "h2d" for span in timeline["spans"])
        self.assertEqual(
            timeline["coverage"],
            {
                "cache_transfer_loads_delta": h2d_span_count,
                "h2d_span_count": h2d_span_count,
            },
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_cuda_timeline_sync_control_has_no_h2d_compute_overlap(self) -> None:
        device = "cuda"
        generator = torch.Generator().manual_seed(921)
        hidden_states = torch.randn(3, self.hidden_size, generator=generator).to(device)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]], device=device)
        top_k_weights = torch.rand(3, 2, generator=generator).to(device)
        cache = self._cache(2, device=device)
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)

        with runtime.cuda_timeline_capture() as capture:
            actual = runtime.forward(
                0,
                hidden_states,
                top_k_index,
                top_k_weights,
            )
            torch.cuda.synchronize()

        torch.testing.assert_close(
            actual,
            self._reference_forward(hidden_states, top_k_index, top_k_weights),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertIsNotNone(capture.result)
        timeline = capture.result
        self.assertEqual(timeline["status"], "measured")
        self.assertEqual(timeline["summary"]["overlap"]["duration_ms"], 0.0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_cuda_pipeline_switch_is_numerically_stable(self) -> None:
        device = "cuda"
        cache = self._cache(
            2,
            device=device,
            staging_slots=2,
            pipeline_mode="async",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        hidden_states = torch.randn(2, self.hidden_size, device=device)
        top_k_weights = torch.ones(2, 1, device=device)

        cache.set_pipeline_mode("sync")
        sync_index = torch.tensor([[0], [1]], device=device)
        sync_output = runtime.forward(0, hidden_states, sync_index, top_k_weights)
        torch.testing.assert_close(
            sync_output,
            self._reference_forward(hidden_states, sync_index, top_k_weights),
            rtol=1e-5,
            atol=1e-6,
        )

        cache.set_pipeline_mode("async")
        async_index = torch.tensor([[2], [3]], device=device)
        async_output = runtime.forward(0, hidden_states, async_index, top_k_weights)
        torch.testing.assert_close(
            async_output,
            self._reference_forward(hidden_states, async_index, top_k_weights),
            rtol=1e-5,
            atol=1e-6,
        )

        cache.set_pipeline_mode("sync")
        retained_output = runtime.forward(0, hidden_states, async_index, top_k_weights)
        torch.testing.assert_close(retained_output, async_output)
        self.assertEqual(cache.pipeline_mode, "sync")
        self.assertEqual((cache.metrics().requests, cache.metrics().hits), (6, 2))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "requires CUDA",
    )
    def test_adaptive_cuda_switches_from_async_fill_to_sync_hit(self) -> None:
        device = "cuda"
        generator = torch.Generator().manual_seed(1509)
        hidden_states = torch.randn(3, self.hidden_size, generator=generator).to(device)
        top_k_index = torch.tensor([[0], [1], [2]], device=device)
        top_k_weights = torch.ones(3, 1, device=device)
        cache = self._cache(
            2,
            device=device,
            staging_slots=2,
            pipeline_mode="adaptive",
        )
        self.addCleanup(cache.close)
        runtime = PagedExpertRuntime(cache)
        expected = self._reference_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        torch.cuda.current_stream().synchronize()
        original_synchronize = torch.cuda.synchronize

        def reject_global_synchronize(*_args, **_kwargs):
            raise AssertionError("adaptive async fill used a global CUDA synchronize")

        torch.cuda.synchronize = reject_global_synchronize
        try:
            actual = runtime.forward(
                0,
                hidden_states,
                top_k_index,
                top_k_weights,
            )
        finally:
            torch.cuda.synchronize = original_synchronize

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        hit_index = torch.tensor([[1], [2]], device=device)
        hit_weights = torch.ones(2, 1, device=device)
        hit_output = runtime.forward(
            0,
            hidden_states[:2],
            hit_index,
            hit_weights,
        )
        torch.testing.assert_close(
            hit_output,
            self._reference_forward(hidden_states[:2], hit_index, hit_weights),
            rtol=1e-5,
            atol=1e-6,
        )
        metrics = cache.metrics()
        self.assertEqual(metrics.adaptive_async_forwards, 1)
        self.assertEqual(metrics.adaptive_sync_forwards, 1)
        self.assertEqual(metrics.adaptive_async_experts, 3)
        self.assertEqual(metrics.adaptive_sync_experts, 2)
        self.assertEqual((metrics.requests, metrics.hits, metrics.misses), (5, 2, 3))


if __name__ == "__main__":
    unittest.main()
