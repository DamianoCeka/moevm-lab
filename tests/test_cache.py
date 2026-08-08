from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
