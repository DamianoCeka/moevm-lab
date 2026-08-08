from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

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

    def put(
        self,
        key: ExpertKey,
        size_bytes: int,
        *,
        protected: Iterable[ExpertKey] = (),
    ) -> tuple[bool, tuple[ExpertKey, ...]]:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.capacity_bytes == 0 or size_bytes > self.capacity_bytes:
            return False, ()

        protected_keys = set(protected)
        previous_size = self._items.get(key)
        used_without_key = self._used_bytes - (previous_size or 0)
        required_bytes = max(0, used_without_key + size_bytes - self.capacity_bytes)

        eviction_plan: list[ExpertKey] = []
        freed_bytes = 0
        if required_bytes:
            for old_key, old_size in self._items.items():
                if old_key == key or old_key in protected_keys:
                    continue
                eviction_plan.append(old_key)
                freed_bytes += old_size
                if freed_bytes >= required_bytes:
                    break
            if freed_bytes < required_bytes:
                return False, ()

        if previous_size is not None:
            self._items.pop(key)
            self._used_bytes -= previous_size

        evicted: list[ExpertKey] = []
        for old_key in eviction_plan:
            old_size = self._items.pop(old_key)
            self._used_bytes -= old_size
            evicted.append(old_key)

        self._items[key] = size_bytes
        self._used_bytes += size_bytes
        return True, tuple(evicted)

    def can_put(
        self,
        key: ExpertKey,
        size_bytes: int,
        *,
        protected: Iterable[ExpertKey] = (),
    ) -> bool:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.capacity_bytes == 0 or size_bytes > self.capacity_bytes:
            return False

        previous_size = self._items.get(key)
        used_without_key = self._used_bytes - (previous_size or 0)
        required_bytes = max(0, used_without_key + size_bytes - self.capacity_bytes)
        if required_bytes == 0:
            return True

        protected_keys = set(protected)
        freed_bytes = 0
        for old_key, old_size in self._items.items():
            if old_key == key or old_key in protected_keys:
                continue
            freed_bytes += old_size
            if freed_bytes >= required_bytes:
                return True
        return False

    def keys(self) -> tuple[ExpertKey, ...]:
        return tuple(self._items.keys())

    def clone(self) -> ByteLRUCache:
        clone = ByteLRUCache(self.capacity_bytes)
        clone._items = self._items.copy()
        clone._used_bytes = self._used_bytes
        return clone


@dataclass(slots=True)
class TransferResult:
    requested: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    storage_hits: int = 0
    bytes_ram_to_vram: int = 0
    bytes_nvme_to_ram: int = 0
    transfer_ms: float = 0.0
    rejected_capacity: int = 0
    l1_hit_keys: set[ExpertKey] = field(default_factory=set)
    loaded_to_l1: set[ExpertKey] = field(default_factory=set)
    resident_loaded_to_l1: set[ExpertKey] = field(default_factory=set)
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

    def _transfer_time_ms(
        self, ram_to_vram_bytes: int, nvme_to_ram_bytes: int
    ) -> float:
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

    def _remove_from_vram(
        self, keys: Iterable[ExpertKey], result: TransferResult
    ) -> None:
        for key in keys:
            removed = self.l1.remove(key)
            removed = self.prefetch_l1.remove(key) or removed
            if removed:
                result.evicted_from_l1.add(key)

    def _clone(self) -> HierarchicalExpertCache:
        clone = object.__new__(HierarchicalExpertCache)
        clone.hardware = self.hardware
        clone.expert_size_bytes = self.expert_size_bytes
        clone.l1 = self.l1.clone()
        clone.prefetch_l1 = self.prefetch_l1.clone()
        clone.l2 = self.l2.clone()
        return clone

    def estimate_transfer_ms(self, keys: Iterable[ExpertKey]) -> float:
        """Estimate an exact prefetch batch cost without touching cache state."""
        simulation = self._clone()
        return simulation.prefetch_many(keys).transfer_ms

    def _estimate_prefetch_bytes(self, key: ExpertKey) -> tuple[int, int]:
        if key in self.l1 or key in self.prefetch_l1:
            return 0, 0
        if self.expert_size_bytes > self.prefetch_l1.capacity_bytes:
            return 0, 0
        if key in self.l2:
            return self.expert_size_bytes, 0
        if not self.l2.can_put(
            key,
            self.expert_size_bytes,
            protected=self.l1.keys(),
        ):
            return 0, 0
        return self.expert_size_bytes, self.expert_size_bytes

    def admit_prefetch_within_budget(
        self,
        keys: Iterable[ExpertKey],
        budget_ms: float,
    ) -> tuple[tuple[ExpertKey, ...], int]:
        """Select the ordered prefix-compatible candidates that meet a batch deadline."""
        if budget_ms < 0:
            raise ValueError("budget_ms cannot be negative")

        simulation = self._clone()
        admitted: list[ExpertKey] = []
        rejected = 0
        ram_to_vram_bytes = 0
        nvme_to_ram_bytes = 0

        for key in self._deduplicate(keys):
            candidate_ram_bytes, candidate_nvme_bytes = (
                simulation._estimate_prefetch_bytes(key)
            )

            proposed_ms = self._transfer_time_ms(
                ram_to_vram_bytes + candidate_ram_bytes,
                nvme_to_ram_bytes + candidate_nvme_bytes,
            )
            if proposed_ms <= budget_ms + 1e-12:
                admitted.append(key)
                ram_to_vram_bytes += candidate_ram_bytes
                nvme_to_ram_bytes += candidate_nvme_bytes
                simulation.prefetch_many((key,))
            else:
                rejected += 1

        return tuple(admitted), rejected

    def access_many(self, keys: Iterable[ExpertKey]) -> TransferResult:
        unique_keys = self._deduplicate(keys)
        result = TransferResult(requested=len(unique_keys))

        for key in unique_keys:
            if self.l1.touch(key):
                result.l1_hits += 1
                result.l1_hit_keys.add(key)
                self.l2.touch(key)
                continue

            if self.prefetch_l1.remove(key):
                result.l1_hits += 1
                result.l1_hit_keys.add(key)
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
                result.l1_hit_keys.add(key)
                self.l2.touch(key)
                continue

            # A speculative transfer has no value if its VRAM partition cannot
            # hold even one expert.
            if self.expert_size_bytes > self.prefetch_l1.capacity_bytes:
                result.rejected_capacity += 1
                continue

            if self.l2.touch(key):
                result.l2_hits += 1
                result.bytes_ram_to_vram += self.expert_size_bytes
            else:
                stored_l2, evicted_l2 = self.l2.put(
                    key,
                    self.expert_size_bytes,
                    protected=self.l1.keys(),
                )
                if not stored_l2:
                    result.rejected_capacity += 1
                    continue
                result.storage_hits += 1
                result.bytes_nvme_to_ram += self.expert_size_bytes
                result.bytes_ram_to_vram += self.expert_size_bytes
                self._remove_from_vram(evicted_l2, result)

            stored, evicted = self.prefetch_l1.put(key, self.expert_size_bytes)
            result.evicted_from_l1.update(evicted)
            if stored:
                result.loaded_to_l1.add(key)

        result.transfer_ms = self._transfer_time_ms(
            result.bytes_ram_to_vram,
            result.bytes_nvme_to_ram,
        )
        result.resident_loaded_to_l1 = result.loaded_to_l1.intersection(
            self.prefetch_l1.keys()
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
