from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

from .config import HardwareConfig
from .types import ExpertKey


class ByteLRUCache:
    """A deterministic byte-capacity LRU cache for logical expert weights."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes cannot be negative")
        self.capacity_bytes = capacity_bytes
        self._items: OrderedDict[ExpertKey, int] = OrderedDict()
        self._used_bytes = 0

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._items

    def touch(self, key: ExpertKey) -> bool:
        if key not in self._items:
            return False
        self._items.move_to_end(key)
        return True

    def remove(self, key: ExpertKey) -> bool:
        size = self._items.pop(key, None)
        if size is None:
            return False
        self._used_bytes -= size
        return True

    def put(self, key: ExpertKey, size_bytes: int) -> tuple[bool, tuple[ExpertKey, ...]]:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.capacity_bytes == 0 or size_bytes > self.capacity_bytes:
            return False, ()

        evicted: list[ExpertKey] = []
        previous_size = self._items.pop(key, None)
        if previous_size is not None:
            self._used_bytes -= previous_size

        while self._items and self._used_bytes + size_bytes > self.capacity_bytes:
            old_key, old_size = self._items.popitem(last=False)
            self._used_bytes -= old_size
            evicted.append(old_key)

        self._items[key] = size_bytes
        self._used_bytes += size_bytes
        return True, tuple(evicted)

    def keys(self) -> tuple[ExpertKey, ...]:
        return tuple(self._items.keys())


@dataclass(slots=True)
class TransferResult:
    requested: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    storage_hits: int = 0
    bytes_ram_to_vram: int = 0
    bytes_nvme_to_ram: int = 0
    transfer_ms: float = 0.0
    loaded_to_l1: set[ExpertKey] = field(default_factory=set)
    evicted_from_l1: set[ExpertKey] = field(default_factory=set)

    @property
    def misses(self) -> int:
        return self.l2_hits + self.storage_hits


class HierarchicalExpertCache:
    """Inclusive VRAM/RAM cache with a protected speculative VRAM partition."""

    def __init__(
        self,
        hardware: HardwareConfig,
        expert_size_bytes: int,
        *,
        prefetch_enabled: bool = False,
    ) -> None:
        if expert_size_bytes <= 0:
            raise ValueError("expert_size_bytes must be positive")
        self.hardware = hardware
        self.expert_size_bytes = expert_size_bytes

        if prefetch_enabled:
            speculative_bytes = int(
                hardware.vram_cache_bytes * hardware.prefetch_vram_fraction
            )
        else:
            speculative_bytes = 0
        demand_bytes = hardware.vram_cache_bytes - speculative_bytes

        self.l1 = ByteLRUCache(demand_bytes)
        self.prefetch_l1 = ByteLRUCache(speculative_bytes)
        self.l2 = ByteLRUCache(hardware.ram_cache_bytes)

    @staticmethod
    def _deduplicate(keys: Iterable[ExpertKey]) -> tuple[ExpertKey, ...]:
        return tuple(dict.fromkeys(keys))

    def _transfer_time_ms(self, ram_to_vram_bytes: int, nvme_to_ram_bytes: int) -> float:
        milliseconds = 0.0
        if nvme_to_ram_bytes:
            milliseconds += (
                nvme_to_ram_bytes / (self.hardware.nvme_to_ram_gbps * 1_000_000_000)
            ) * 1000.0
            milliseconds += self.hardware.nvme_latency_us / 1000.0
        if ram_to_vram_bytes:
            milliseconds += (
                ram_to_vram_bytes / (self.hardware.ram_to_vram_gbps * 1_000_000_000)
            ) * 1000.0
            milliseconds += self.hardware.ram_latency_us / 1000.0
        return milliseconds

    def _remove_from_vram(self, keys: Iterable[ExpertKey], result: TransferResult) -> None:
        for key in keys:
            removed = self.l1.remove(key)
            removed = self.prefetch_l1.remove(key) or removed
            if removed:
                result.evicted_from_l1.add(key)

    def estimate_transfer_ms(self, keys: Iterable[ExpertKey]) -> float:
        """Estimate current transfer cost without touching cache state."""
        ram_to_vram_bytes = 0
        nvme_to_ram_bytes = 0
        for key in self._deduplicate(keys):
            if key in self.l1 or key in self.prefetch_l1:
                continue
            ram_to_vram_bytes += self.expert_size_bytes
            if key not in self.l2:
                nvme_to_ram_bytes += self.expert_size_bytes
        return self._transfer_time_ms(ram_to_vram_bytes, nvme_to_ram_bytes)

    def access_many(self, keys: Iterable[ExpertKey]) -> TransferResult:
        unique_keys = self._deduplicate(keys)
        result = TransferResult(requested=len(unique_keys))

        for key in unique_keys:
            if self.l1.touch(key):
                result.l1_hits += 1
                self.l2.touch(key)
                continue

            if self.prefetch_l1.remove(key):
                result.l1_hits += 1
                self.l2.touch(key)
                _, evicted_l1 = self.l1.put(key, self.expert_size_bytes)
                result.evicted_from_l1.update(evicted_l1)
                continue

            if self.l2.touch(key):
                result.l2_hits += 1
                result.bytes_ram_to_vram += self.expert_size_bytes
            else:
                result.storage_hits += 1
                result.bytes_nvme_to_ram += self.expert_size_bytes
                result.bytes_ram_to_vram += self.expert_size_bytes
                stored_l2, evicted_l2 = self.l2.put(key, self.expert_size_bytes)
                if stored_l2:
                    self._remove_from_vram(evicted_l2, result)

            stored_l1, evicted_l1 = self.l1.put(key, self.expert_size_bytes)
            result.evicted_from_l1.update(evicted_l1)
            if stored_l1:
                result.loaded_to_l1.add(key)

        result.transfer_ms = self._transfer_time_ms(
            result.bytes_ram_to_vram,
            result.bytes_nvme_to_ram,
        )
        return result

    def prefetch_many(self, keys: Iterable[ExpertKey]) -> TransferResult:
        unique_keys = self._deduplicate(keys)
        result = TransferResult(requested=len(unique_keys))

        for key in unique_keys:
            if self.l1.touch(key) or self.prefetch_l1.touch(key):
                result.l1_hits += 1
                self.l2.touch(key)
                continue

            if self.l2.touch(key):
                result.l2_hits += 1
                result.bytes_ram_to_vram += self.expert_size_bytes
            else:
                result.storage_hits += 1
                result.bytes_nvme_to_ram += self.expert_size_bytes
                result.bytes_ram_to_vram += self.expert_size_bytes
                stored_l2, evicted_l2 = self.l2.put(key, self.expert_size_bytes)
                if stored_l2:
                    self._remove_from_vram(evicted_l2, result)

            stored, evicted = self.prefetch_l1.put(key, self.expert_size_bytes)
            result.evicted_from_l1.update(evicted)
            if stored:
                result.loaded_to_l1.add(key)

        result.transfer_ms = self._transfer_time_ms(
            result.bytes_ram_to_vram,
            result.bytes_nvme_to_ram,
        )
        return result

    def snapshot(self) -> dict[str, int]:
        return {
            "vram_demand_items": len(self.l1),
            "vram_demand_used_bytes": self.l1.used_bytes,
            "vram_demand_capacity_bytes": self.l1.capacity_bytes,
            "vram_prefetch_items": len(self.prefetch_l1),
            "vram_prefetch_used_bytes": self.prefetch_l1.used_bytes,
            "vram_prefetch_capacity_bytes": self.prefetch_l1.capacity_bytes,
            "ram_items": len(self.l2),
            "ram_used_bytes": self.l2.used_bytes,
            "ram_capacity_bytes": self.l2.capacity_bytes,
        }
