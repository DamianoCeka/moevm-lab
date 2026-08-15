from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as torch_functional
    from safetensors.torch import save_file
except ImportError:
    torch = None

if torch is not None:
    from moevm.paged_runtime import (
        CachePolicy,
        ExpertSlotCache,
        PagedExpertRuntime,
        SafetensorExpertStore,
        _cuda_timeline_payload,
        _CudaTimelineEventSpan,
        attach_transformers_olmoe_runtime,
        load_non_expert_weights_into_meta_model,
        register_transformers_paged_experts,
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
        self.assertEqual(payload["unit"], "milliseconds")
        self.assertEqual(payload["spans"][0]["name"], "h2d:0:L0:E1")
        self.assertEqual(payload["spans"][1]["name"], "expert_compute:1:L0:E2")
        self.assertEqual(payload["summary"]["overlap"]["duration_ms"], 2.0)

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
        self.assertGreater(timeline["summary"]["transfer"]["interval_count"], 0)
        self.assertGreater(timeline["summary"]["compute"]["interval_count"], 0)
        self.assertGreaterEqual(timeline["summary"]["overlap"]["duration_ms"], 0.0)
        self.assertTrue(all(span["layer"] == 0 for span in timeline["spans"]))

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
