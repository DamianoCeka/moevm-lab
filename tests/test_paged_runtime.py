from __future__ import annotations

import json
import tempfile
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
    ) -> ExpertSlotCache:
        return ExpertSlotCache(
            self.store,
            capacity=capacity,
            device=device,
            policy=policy,
            static_keys=static_keys,
            staging_slots=1,
            pin_staging=device == "cuda",
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


if __name__ == "__main__":
    unittest.main()
