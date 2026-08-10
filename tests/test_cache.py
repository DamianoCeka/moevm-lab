from __future__ import annotations

import unittest
from dataclasses import replace

from moevm.cache import ByteLRUCache, HierarchicalExpertCache
from moevm.config import HardwareConfig
from moevm.types import ExpertKey


class ByteLRUCacheTests(unittest.TestCase):
    def test_evicts_least_recently_used_item(self) -> None:
        cache = ByteLRUCache(capacity_bytes=20)
        a = ExpertKey(0, 0)
        b = ExpertKey(0, 1)
        c = ExpertKey(0, 2)

        cache.put(a, 10)
        cache.put(b, 10)
        cache.touch(a)
        stored, evicted = cache.put(c, 10)

        self.assertTrue(stored)
        self.assertEqual(evicted, (b,))
        self.assertIn(a, cache)
        self.assertIn(c, cache)
        self.assertNotIn(b, cache)

    def test_rejects_item_larger_than_capacity(self) -> None:
        cache = ByteLRUCache(capacity_bytes=8)
        stored, evicted = cache.put(ExpertKey(0, 0), 9)
        self.assertFalse(stored)
        self.assertEqual(evicted, ())
        self.assertEqual(len(cache), 0)


class HierarchicalCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = HardwareConfig(
            vram_cache_mib=2.0,
            ram_cache_mib=4.0,
            ram_to_vram_gbps=20.0,
            nvme_to_ram_gbps=5.0,
            ram_latency_us=10.0,
            nvme_latency_us=100.0,
            overlap_efficiency=0.8,
            prefetch_vram_fraction=0.25,
        )

    def test_first_access_is_storage_then_vram_hit(self) -> None:
        cache = HierarchicalExpertCache(self.hardware, expert_size_bytes=1024 * 1024)
        key = ExpertKey(0, 7)

        first = cache.access_many([key])
        second = cache.access_many([key])

        self.assertEqual(first.storage_hits, 1)
        self.assertGreater(first.bytes_nvme_to_ram, 0)
        self.assertEqual(second.l1_hits, 1)
        self.assertEqual(second.transfer_ms, 0.0)

    def test_batch_estimate_accounts_for_intra_batch_l2_eviction(self) -> None:
        hardware = HardwareConfig(
            vram_cache_mib=4.0,
            ram_cache_mib=1.0,
            ram_to_vram_gbps=10.0,
            nvme_to_ram_gbps=10.0,
            ram_latency_us=0.0,
            nvme_latency_us=0.0,
            overlap_efficiency=1.0,
            prefetch_vram_fraction=0.5,
        )
        cache = HierarchicalExpertCache(
            hardware,
            expert_size_bytes=1024 * 1024,
            prefetch_enabled=True,
        )
        a = ExpertKey(0, 0)
        b = ExpertKey(0, 1)
        cache.l2.put(b, 1024 * 1024)

        estimated = cache.estimate_transfer_ms((a, b))
        actual = cache.prefetch_many((a, b)).transfer_ms

        self.assertAlmostEqual(estimated, actual)
        self.assertEqual(cache.prefetch_many(()).transfer_ms, 0.0)

    def test_per_expert_latency_counts_mixed_tier_transfers(self) -> None:
        hardware = replace(
            self.hardware,
            vram_cache_mib=4.0,
            fixed_latency_scope="per_expert",
        )
        expert_size = 1024 * 1024
        cache = HierarchicalExpertCache(hardware, expert_size_bytes=expert_size)
        storage_key = ExpertKey(0, 0)
        ram_key = ExpertKey(0, 1)
        cache.l2.put(ram_key, expert_size)

        result = cache.access_many((storage_key, ram_key, storage_key))

        self.assertEqual(result.ram_to_vram_transfers, 2)
        self.assertEqual(result.nvme_to_ram_transfers, 1)
        expected_ms = (
            (2 * expert_size / (hardware.ram_to_vram_gbps * 1_000_000_000)) * 1000.0
            + 2 * hardware.ram_latency_us / 1000.0
            + (expert_size / (hardware.nvme_to_ram_gbps * 1_000_000_000)) * 1000.0
            + hardware.nvme_latency_us / 1000.0
        )
        self.assertAlmostEqual(result.transfer_ms, expected_ms)

    def test_latency_scope_changes_deadline_admission(self) -> None:
        hardware = HardwareConfig(
            vram_cache_mib=4.0,
            ram_cache_mib=4.0,
            ram_to_vram_gbps=1000.0,
            nvme_to_ram_gbps=1000.0,
            ram_latency_us=600.0,
            nvme_latency_us=0.0,
            overlap_efficiency=1.0,
            prefetch_vram_fraction=0.5,
            fixed_latency_scope="per_expert",
        )
        expert_size = 1024 * 1024
        first = ExpertKey(0, 0)
        second = ExpertKey(0, 1)

        per_expert = HierarchicalExpertCache(
            hardware,
            expert_size_bytes=expert_size,
            prefetch_enabled=True,
        )
        per_expert.l2.put(first, expert_size)
        per_expert.l2.put(second, expert_size)
        admitted, rejected = per_expert.admit_prefetch_within_budget(
            (first, second),
            budget_ms=0.9,
        )

        self.assertEqual(admitted, (first,))
        self.assertEqual(rejected, 1)

        batch = HierarchicalExpertCache(
            replace(hardware, fixed_latency_scope="batch"),
            expert_size_bytes=expert_size,
            prefetch_enabled=True,
        )
        batch.l2.put(first, expert_size)
        batch.l2.put(second, expert_size)
        admitted, rejected = batch.admit_prefetch_within_budget(
            (first, second),
            budget_ms=0.9,
        )

        self.assertEqual(admitted, (first, second))
        self.assertEqual(rejected, 0)

    def test_speculative_l2_admission_preserves_demand_vram(self) -> None:
        hardware = HardwareConfig(
            vram_cache_mib=2.0,
            ram_cache_mib=2.0,
            ram_to_vram_gbps=10.0,
            nvme_to_ram_gbps=10.0,
            ram_latency_us=0.0,
            nvme_latency_us=0.0,
            overlap_efficiency=1.0,
            prefetch_vram_fraction=0.5,
        )
        cache = HierarchicalExpertCache(
            hardware,
            expert_size_bytes=1024 * 1024,
            prefetch_enabled=True,
        )
        demand = ExpertKey(0, 0)
        first_speculative = ExpertKey(0, 1)
        second_speculative = ExpertKey(0, 2)

        cache.access_many((demand,))
        cache.prefetch_many((first_speculative,))
        cache.prefetch_many((second_speculative,))

        self.assertIn(demand, cache.l1)
        self.assertIn(demand, cache.l2)

    def test_zero_capacity_speculative_buffer_skips_transfer(self) -> None:
        cache = HierarchicalExpertCache(
            self.hardware,
            expert_size_bytes=1024 * 1024,
            prefetch_enabled=False,
        )

        result = cache.prefetch_many((ExpertKey(0, 9),))

        self.assertEqual(result.bytes_nvme_to_ram, 0)
        self.assertEqual(result.bytes_ram_to_vram, 0)
        self.assertEqual(result.loaded_to_l1, set())
        self.assertEqual(result.rejected_capacity, 1)

    def test_prefetch_result_distinguishes_reloaded_final_resident(self) -> None:
        hardware = HardwareConfig(
            vram_cache_mib=2.0,
            ram_cache_mib=4.0,
            ram_to_vram_gbps=10.0,
            nvme_to_ram_gbps=10.0,
            ram_latency_us=0.0,
            nvme_latency_us=0.0,
            overlap_efficiency=1.0,
            prefetch_vram_fraction=0.5,
        )
        cache = HierarchicalExpertCache(
            hardware,
            expert_size_bytes=1024 * 1024,
            prefetch_enabled=True,
        )
        a = ExpertKey(0, 0)
        b = ExpertKey(0, 1)
        cache.prefetch_many((a,))

        result = cache.prefetch_many((b, a))

        self.assertEqual(result.loaded_to_l1, {a, b})
        self.assertEqual(result.evicted_from_l1, {a, b})
        self.assertEqual(result.resident_loaded_to_l1, {a})
        self.assertIn(a, cache.prefetch_l1)
        self.assertNotIn(b, cache.prefetch_l1)


if __name__ == "__main__":
    unittest.main()
