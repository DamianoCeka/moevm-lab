"""Bounded, read-only expert paging primitives for real OLMoE checkpoints.

This module deliberately keeps the first hardware runtime small and synchronous:
only the experts selected by the router are read, staging memory is preallocated,
and every cache slot has a fixed allocation.  It is an inference prototype, not a
training implementation.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self

try:
    import torch
    import torch.nn.functional as torch_functional
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "moevm.paged_runtime requires the 'real-traces' optional dependencies"
    ) from exc

from .types import ExpertKey

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


@dataclass(frozen=True, slots=True)
class ExpertSpec:
    """Packed OLMoE expert layout used by Transformers."""

    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("expert dimensions must be positive")

    @property
    def gate_up_shape(self) -> tuple[int, int]:
        return (2 * self.intermediate_size, self.hidden_size)

    @property
    def down_shape(self) -> tuple[int, int]:
        return (self.hidden_size, self.intermediate_size)

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def size_bytes(self) -> int:
        elements = 3 * self.hidden_size * self.intermediate_size
        return elements * self.element_size


@dataclass(frozen=True, slots=True)
class ExpertWeights:
    """One expert in the packed gate/up plus down layout."""

    gate_up: torch.Tensor
    down: torch.Tensor


class SafetensorExpertStore:
    """Read-only, index-driven access to per-expert safetensors weights.

    The Hugging Face index is parsed once.  Shards are opened only while a
    requested tensor is copied; the store never rewrites or downloads files.
    """

    def __init__(
        self,
        snapshot_or_index: str | Path,
        *,
        tensor_prefix: str = "model.layers",
    ) -> None:
        supplied = Path(snapshot_or_index).expanduser()
        if supplied.is_dir():
            index_path = supplied / "model.safetensors.index.json"
        else:
            index_path = supplied
        if not index_path.is_file():
            raise FileNotFoundError(f"safetensors index not found: {index_path}")

        self.index_path = index_path.resolve()
        self.snapshot_path = self.index_path.parent
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index must contain a non-empty weight_map")
        if not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise ValueError("safetensors weight_map keys and values must be strings")
        self._weight_map: dict[str, str] = dict(weight_map)

        escaped_prefix = re.escape(tensor_prefix)
        self._expert_pattern = re.compile(
            rf"^{escaped_prefix}\.(\d+)\.mlp\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.weight$"
        )
        entries: dict[ExpertKey, dict[str, str]] = {}
        for tensor_name in self._weight_map:
            match = self._expert_pattern.fullmatch(tensor_name)
            if match is None:
                continue
            key = ExpertKey(int(match.group(1)), int(match.group(2)))
            projection = match.group(3)
            entries.setdefault(key, {})[projection] = tensor_name

        if not entries:
            raise ValueError("index contains no OLMoE per-expert weights")
        incomplete = {
            key: sorted(set(_PROJECTIONS) - projections.keys())
            for key, projections in entries.items()
            if projections.keys() != set(_PROJECTIONS)
        }
        if incomplete:
            first_key = min(incomplete)
            missing = ", ".join(incomplete[first_key])
            raise ValueError(f"incomplete weights for {first_key.compact()}: {missing}")

        self._entries = entries
        self._validate_shard_paths()
        self._handle_lock = threading.RLock()
        self._handles: dict[Path, Any] = {}
        try:
            for shard_name in sorted(set(self._weight_map.values())):
                shard_path = (self.snapshot_path / shard_name).resolve()
                handle = safe_open(shard_path, framework="pt", device="cpu")
                handle.__enter__()
                self._handles[shard_path] = handle
        except Exception:
            self.close()
            raise
        self.spec = self._read_and_validate_spec(min(self._entries))

    def _shard_path(self, tensor_name: str) -> Path:
        try:
            shard_name = self._weight_map[tensor_name]
        except KeyError as exc:
            raise KeyError(f"tensor not present in index: {tensor_name}") from exc
        shard_path = (self.snapshot_path / shard_name).resolve()
        if not shard_path.is_relative_to(self.snapshot_path):
            raise ValueError(
                f"safetensors shard escapes snapshot directory: {shard_name}"
            )
        return shard_path

    def _validate_shard_paths(self) -> None:
        for shard_name in sorted(set(self._weight_map.values())):
            shard_path = (self.snapshot_path / shard_name).resolve()
            if not shard_path.is_relative_to(self.snapshot_path):
                raise ValueError(
                    f"safetensors shard escapes snapshot directory: {shard_name}"
                )
            if not shard_path.is_file():
                raise FileNotFoundError(f"safetensors shard not found: {shard_path}")

    @staticmethod
    def _torch_dtype(safetensors_dtype: str) -> torch.dtype:
        try:
            return _SAFETENSORS_DTYPES[safetensors_dtype]
        except KeyError as exc:
            raise ValueError(
                f"unsupported safetensors dtype: {safetensors_dtype}"
            ) from exc

    def _tensor_metadata(self, tensor_name: str) -> tuple[tuple[int, ...], torch.dtype]:
        shard_path = self._shard_path(tensor_name)
        with self._handle_lock:
            handle = self._open_handle(shard_path)
            tensor_slice = handle.get_slice(tensor_name)
            shape = tuple(tensor_slice.get_shape())
            dtype = self._torch_dtype(tensor_slice.get_dtype())
        return shape, dtype

    def _open_handle(self, shard_path: Path) -> Any:
        try:
            return self._handles[shard_path]
        except KeyError as exc:
            raise RuntimeError("safetensors expert store is closed") from exc

    def close(self) -> None:
        """Close all persistent read-only shard mappings."""
        with getattr(self, "_handle_lock", threading.RLock()):
            handles = getattr(self, "_handles", {})
            for handle in handles.values():
                handle.__exit__(None, None, None)
            handles.clear()

    def __enter__(self) -> Self:
        if not self._handles:
            raise RuntimeError("safetensors expert store is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_and_validate_spec(self, key: ExpertKey) -> ExpertSpec:
        names = self._entries[key]
        gate_shape, gate_dtype = self._tensor_metadata(names["gate_proj"])
        up_shape, up_dtype = self._tensor_metadata(names["up_proj"])
        down_shape, down_dtype = self._tensor_metadata(names["down_proj"])
        if len(gate_shape) != 2:
            raise ValueError(f"gate projection for {key.compact()} must be rank 2")
        intermediate_size, hidden_size = gate_shape
        spec = ExpertSpec(hidden_size, intermediate_size, gate_dtype)
        if up_shape != gate_shape or down_shape != spec.down_shape:
            raise ValueError(f"incompatible projection shapes for {key.compact()}")
        if up_dtype != spec.dtype or down_dtype != spec.dtype:
            raise ValueError(f"mixed projection dtypes for {key.compact()}")
        return spec

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def keys(self) -> tuple[ExpertKey, ...]:
        return tuple(sorted(self._entries))

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({key.layer for key in self._entries}))

    def experts_in_layer(self, layer: int) -> tuple[int, ...]:
        return tuple(key.expert for key in self.keys if key.layer == layer)

    def load_into(
        self,
        key: ExpertKey,
        gate_up_destination: torch.Tensor,
        down_destination: torch.Tensor,
    ) -> int:
        """Copy one expert into existing CPU buffers and return bytes read."""
        if key not in self._entries:
            raise KeyError(f"unknown expert: {key.compact()}")
        if (
            gate_up_destination.device.type != "cpu"
            or down_destination.device.type != "cpu"
        ):
            raise ValueError("safetensors destinations must be CPU tensors")
        if gate_up_destination.shape != self.spec.gate_up_shape:
            raise ValueError("gate/up destination has the wrong shape")
        if down_destination.shape != self.spec.down_shape:
            raise ValueError("down destination has the wrong shape")
        if (
            gate_up_destination.dtype != self.spec.dtype
            or down_destination.dtype != self.spec.dtype
        ):
            raise ValueError("destination dtype does not match the checkpoint")

        destinations = {
            "gate_proj": gate_up_destination[: self.spec.intermediate_size],
            "up_proj": gate_up_destination[self.spec.intermediate_size :],
            "down_proj": down_destination,
        }
        names = self._entries[key]
        names_by_shard: dict[Path, list[tuple[str, str]]] = {}
        for projection in _PROJECTIONS:
            tensor_name = names[projection]
            names_by_shard.setdefault(self._shard_path(tensor_name), []).append(
                (projection, tensor_name)
            )

        bytes_read = 0
        with self._handle_lock:
            for shard_path, projected_names in names_by_shard.items():
                handle = self._open_handle(shard_path)
                for projection, tensor_name in projected_names:
                    source = handle.get_tensor(tensor_name)
                    destination = destinations[projection]
                    if (
                        source.shape != destination.shape
                        or source.dtype != destination.dtype
                    ):
                        raise ValueError(
                            f"projection metadata changed for {key.compact()}:{projection}"
                        )
                    destination.copy_(source)
                    bytes_read += source.numel() * source.element_size()
        return bytes_read

    def load(self, key: ExpertKey, *, pin_memory: bool = False) -> ExpertWeights:
        gate_up = torch.empty(
            self.spec.gate_up_shape,
            dtype=self.spec.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        down = torch.empty(
            self.spec.down_shape,
            dtype=self.spec.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.load_into(key, gate_up, down)
        return ExpertWeights(gate_up, down)

    def is_expert_tensor(self, tensor_name: str) -> bool:
        return self._expert_pattern.fullmatch(tensor_name) is not None

    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load one arbitrary indexed tensor without mutating the snapshot."""
        shard_path = self._shard_path(tensor_name)
        with self._handle_lock:
            handle = self._open_handle(shard_path)
            return handle.get_tensor(tensor_name)

    def iter_non_expert_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield non-expert tensors one at a time for a bounded meta-model loader."""
        for tensor_name in sorted(self._weight_map):
            if not self.is_expert_tensor(tensor_name):
                yield tensor_name, self.load_tensor(tensor_name)


class CachePolicy(str, Enum):
    """Deterministic expert residency policies."""

    LRU = "lru"
    STATIC = "static"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class PagedRuntimeMetrics:
    requests: int
    hits: int
    misses: int
    evictions: int
    storage_bytes: int
    host_to_device_bytes: int
    storage_seconds: float
    transfer_seconds: float
    forward_seconds: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0


@dataclass(slots=True)
class _MutableMetrics:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    storage_bytes: int = 0
    host_to_device_bytes: int = 0
    storage_seconds: float = 0.0
    transfer_seconds: float = 0.0
    forward_seconds: float = 0.0

    def snapshot(self) -> PagedRuntimeMetrics:
        return PagedRuntimeMetrics(
            requests=self.requests,
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            storage_bytes=self.storage_bytes,
            host_to_device_bytes=self.host_to_device_bytes,
            storage_seconds=self.storage_seconds,
            transfer_seconds=self.transfer_seconds,
            forward_seconds=self.forward_seconds,
        )


class ExpertSlotCache:
    """Fixed-allocation cache of expert weights on a target device.

    Static keys own the first slots.  ``static`` uses one deterministic
    transient slot for every other expert, ``hybrid`` manages all remaining
    slots with LRU, and ``lru`` manages every slot with LRU.  ``capacity`` is
    one global pool; ``capacity_per_layer`` creates independent equal-sized
    partitions and is the mode matching per-layer placement studies.
    """

    def __init__(
        self,
        store: SafetensorExpertStore,
        *,
        capacity: int | None = None,
        capacity_per_layer: int | None = None,
        device: str | torch.device,
        policy: CachePolicy | str = CachePolicy.LRU,
        static_keys: Iterable[ExpertKey] = (),
        staging_slots: int = 1,
        pin_staging: bool | None = None,
    ) -> None:
        if (capacity is None) == (capacity_per_layer is None):
            raise ValueError("set exactly one of capacity or capacity_per_layer")
        selected_capacity = capacity if capacity is not None else capacity_per_layer
        if selected_capacity is None or selected_capacity <= 0:
            raise ValueError("capacity must be positive")
        if staging_slots <= 0:
            raise ValueError("staging_slots must be positive")
        self.store = store
        self.capacity_per_layer = capacity_per_layer
        if capacity_per_layer is None:
            self.capacity = selected_capacity
        else:
            self.capacity = selected_capacity * len(store.layers)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.policy = CachePolicy(policy)
        static_tuple = tuple(dict.fromkeys(static_keys))
        if self.policy is CachePolicy.LRU and static_tuple:
            raise ValueError("static_keys are not valid with the LRU policy")
        unknown = [key for key in static_tuple if key not in store]
        if unknown:
            raise KeyError(f"unknown static expert: {unknown[0].compact()}")
        if capacity_per_layer is None and len(static_tuple) > self.capacity:
            raise ValueError("static expert count exceeds cache capacity")
        self.static_keys = static_tuple
        self._static_slots: dict[ExpertKey, int] = {}
        self._layer_slots: dict[int, tuple[int, ...]] = {}
        self._dynamic_slots_by_layer: dict[int, tuple[int, ...]] = {}
        if capacity_per_layer is None:
            all_slots = tuple(range(self.capacity))
            self._static_slots = {key: slot for slot, key in enumerate(static_tuple)}
            dynamic_slots = tuple(range(len(static_tuple), self.capacity))
            for layer in store.layers:
                self._layer_slots[layer] = all_slots
                self._dynamic_slots_by_layer[layer] = dynamic_slots
            if (
                self.policy is not CachePolicy.LRU
                and len(store) > len(static_tuple)
                and not dynamic_slots
            ):
                raise ValueError(
                    "static and hybrid policies need at least one dynamic slot"
                )
        else:
            static_by_layer = {
                layer: tuple(key for key in static_tuple if key.layer == layer)
                for layer in store.layers
            }
            for layer_offset, layer in enumerate(store.layers):
                layer_start = layer_offset * capacity_per_layer
                layer_slots = tuple(
                    range(layer_start, layer_start + capacity_per_layer)
                )
                self._layer_slots[layer] = layer_slots
                layer_static = static_by_layer[layer]
                if len(layer_static) > capacity_per_layer:
                    raise ValueError(
                        f"static expert count exceeds capacity for layer {layer}"
                    )
                for local_slot, key in enumerate(layer_static):
                    self._static_slots[key] = layer_start + local_slot
                dynamic_slots = layer_slots[len(layer_static) :]
                self._dynamic_slots_by_layer[layer] = dynamic_slots
                remaining_keys = len(store.experts_in_layer(layer)) - len(layer_static)
                if (
                    self.policy is not CachePolicy.LRU
                    and remaining_keys
                    and not dynamic_slots
                ):
                    raise ValueError(
                        "static and hybrid policies need at least one dynamic "
                        f"slot in layer {layer}"
                    )

        spec = store.spec
        self._gate_up_slots = torch.empty(
            (self.capacity, *spec.gate_up_shape),
            dtype=spec.dtype,
            device=self.device,
        )
        self._down_slots = torch.empty(
            (self.capacity, *spec.down_shape),
            dtype=spec.dtype,
            device=self.device,
        )

        if pin_staging is None:
            pin_staging = self.device.type == "cuda"
        if pin_staging and not torch.cuda.is_available():
            raise RuntimeError(
                "pinned staging requested without an available CUDA device"
            )
        self.pin_staging = pin_staging
        self.staging_slots = staging_slots
        self._staging_gate_up = torch.empty(
            (staging_slots, *spec.gate_up_shape),
            dtype=spec.dtype,
            device="cpu",
            pin_memory=pin_staging,
        )
        self._staging_down = torch.empty(
            (staging_slots, *spec.down_shape),
            dtype=spec.dtype,
            device="cpu",
            pin_memory=pin_staging,
        )

        self._key_to_slot: dict[ExpertKey, int] = {}
        self._slot_to_key: list[ExpertKey | None] = [None] * self.capacity
        self._last_used: list[int] = [0] * self.capacity
        self._clock = 0
        self._next_staging_slot = 0
        self._metrics = _MutableMetrics()
        self._lock = threading.RLock()
        self.execution_lock = threading.RLock()

    @property
    def allocated_cache_bytes(self) -> int:
        return self.capacity * self.store.spec.size_bytes

    @property
    def allocated_staging_bytes(self) -> int:
        return self.staging_slots * self.store.spec.size_bytes

    @property
    def resident_keys(self) -> tuple[ExpertKey, ...]:
        with self._lock:
            return tuple(sorted(self._key_to_slot))

    def _select_lru_slot(self, slots: tuple[int, ...]) -> int:
        empty = [slot for slot in slots if self._slot_to_key[slot] is None]
        if empty:
            return min(empty)
        return min(slots, key=lambda slot: (self._last_used[slot], slot))

    def _select_slot(self, key: ExpertKey) -> int:
        static_slot = self._static_slots.get(key)
        if static_slot is not None:
            return static_slot
        dynamic_slots = self._dynamic_slots_by_layer[key.layer]
        if self.policy is CachePolicy.STATIC:
            return dynamic_slots[0]
        if self.policy is CachePolicy.HYBRID:
            return self._select_lru_slot(dynamic_slots)
        return self._select_lru_slot(self._layer_slots[key.layer])

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _copy_staging_to_slot(
        self,
        slot: int,
        staging_gate_up: torch.Tensor,
        staging_down: torch.Tensor,
    ) -> None:
        self._gate_up_slots[slot].copy_(staging_gate_up, non_blocking=False)
        self._down_slots[slot].copy_(staging_down, non_blocking=False)
        self._synchronize_device()

    def get(self, key: ExpertKey) -> ExpertWeights:
        """Return weights valid until a later miss reuses their cache slot."""
        if key not in self.store:
            raise KeyError(f"unknown expert: {key.compact()}")
        with self.execution_lock, self._lock:
            self._clock += 1
            self._metrics.requests += 1
            existing_slot = self._key_to_slot.get(key)
            if existing_slot is not None:
                self._last_used[existing_slot] = self._clock
                self._metrics.hits += 1
                return ExpertWeights(
                    self._gate_up_slots[existing_slot],
                    self._down_slots[existing_slot],
                )

            self._metrics.misses += 1
            slot = self._select_slot(key)
            evicted_key = self._slot_to_key[slot]

            staging_slot = self._next_staging_slot
            self._next_staging_slot = (staging_slot + 1) % self.staging_slots
            staging_gate_up = self._staging_gate_up[staging_slot]
            staging_down = self._staging_down[staging_slot]

            storage_started = time.perf_counter()
            try:
                bytes_read = self.store.load_into(
                    key,
                    staging_gate_up,
                    staging_down,
                )
            finally:
                self._metrics.storage_seconds += time.perf_counter() - storage_started
            self._metrics.storage_bytes += bytes_read

            if evicted_key is not None:
                self._synchronize_device()
                del self._key_to_slot[evicted_key]
                self._slot_to_key[slot] = None
                self._last_used[slot] = 0
                self._metrics.evictions += 1

            transfer_started = time.perf_counter()
            try:
                self._copy_staging_to_slot(slot, staging_gate_up, staging_down)
            except BaseException:
                self._slot_to_key[slot] = None
                self._last_used[slot] = 0
                raise
            finally:
                self._metrics.transfer_seconds += time.perf_counter() - transfer_started
            if self.device.type != "cpu":
                self._metrics.host_to_device_bytes += self.store.spec.size_bytes

            self._slot_to_key[slot] = key
            self._key_to_slot[key] = slot
            self._last_used[slot] = self._clock
            return ExpertWeights(
                self._gate_up_slots[slot],
                self._down_slots[slot],
            )

    def prefetch(self, keys: Iterable[ExpertKey]) -> None:
        with self.execution_lock:
            for key in dict.fromkeys(keys):
                self.get(key)

    def metrics(self) -> PagedRuntimeMetrics:
        with self._lock:
            return self._metrics.snapshot()

    def add_forward_seconds(self, elapsed: float) -> None:
        with self._lock:
            self._metrics.forward_seconds += elapsed


class PagedExpertRuntime:
    """Execute routed OLMoE experts through a bounded slot cache."""

    def __init__(self, cache: ExpertSlotCache) -> None:
        self.cache = cache
        self._forward_lock = cache.execution_lock

    def forward(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        with self._forward_lock:
            return self._forward_locked(
                layer,
                hidden_states,
                top_k_index,
                top_k_weights,
            )

    def _forward_locked(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if layer not in self.cache.store.layers:
            raise KeyError(f"unknown expert layer: {layer}")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden]")
        if top_k_index.ndim != 2 or top_k_weights.shape != top_k_index.shape:
            raise ValueError(
                "routing tensors must have matching [tokens, top_k] shapes"
            )
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError("routing token count must match hidden_states")
        if hidden_states.shape[1] != self.cache.store.spec.hidden_size:
            raise ValueError("hidden_states width does not match expert weights")
        if hidden_states.device != self.cache.device:
            raise ValueError("hidden_states must be on the expert cache device")
        if hidden_states.dtype != self.cache.store.spec.dtype:
            raise ValueError("hidden_states dtype does not match expert weights")
        if (
            top_k_index.device != hidden_states.device
            or top_k_weights.device != hidden_states.device
        ):
            raise ValueError("routing tensors must be on the hidden_states device")

        started = time.perf_counter()
        final_hidden_states = torch.zeros_like(hidden_states)
        active_experts = torch.unique(top_k_index.detach(), sorted=True).cpu().tolist()
        valid_experts = set(self.cache.store.experts_in_layer(layer))
        sentinel = max(valid_experts, default=-1) + 1
        for expert in active_experts:
            expert_id = int(expert)
            if expert_id == sentinel:
                continue
            if expert_id not in valid_experts:
                raise KeyError(f"unknown expert: L{layer}:E{expert_id}")
            weights = self.cache.get(ExpertKey(layer, expert_id))
            token_index, top_k_position = torch.where(top_k_index == expert_id)
            current_state = hidden_states[token_index]
            gate, up = torch_functional.linear(current_state, weights.gate_up).chunk(
                2, dim=-1
            )
            current = torch_functional.silu(gate) * up
            current = torch_functional.linear(current, weights.down)
            current = current * top_k_weights[token_index, top_k_position, None]
            final_hidden_states.index_add_(
                0, token_index, current.to(final_hidden_states.dtype)
            )
        if hidden_states.device.type == "cuda":
            torch.cuda.synchronize(hidden_states.device)
        self.cache.add_forward_seconds(time.perf_counter() - started)
        return final_hidden_states

    def metrics(self) -> PagedRuntimeMetrics:
        return self.cache.metrics()


def transformers_paged_experts_forward(
    expert_module: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Forward registered with Transformers' ``ALL_EXPERTS_FUNCTIONS``."""
    runtime = getattr(expert_module, "_moevm_paged_runtime", None)
    layer = getattr(expert_module, "_moevm_layer_index", None)
    if not isinstance(runtime, PagedExpertRuntime) or not isinstance(layer, int):
        raise RuntimeError("paged expert module is not attached to a MoEVM runtime")
    return runtime.forward(layer, hidden_states, top_k_index, top_k_weights)


def register_transformers_paged_experts(
    implementation: str = "moevm_paged",
) -> None:
    """Register the runtime with a compatible Transformers installation."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    existing = ALL_EXPERTS_FUNCTIONS.get(implementation)
    if existing is not None and existing is not transformers_paged_experts_forward:
        raise ValueError(
            f"Transformers expert implementation already exists: {implementation}"
        )
    ALL_EXPERTS_FUNCTIONS.register(
        implementation,
        transformers_paged_experts_forward,
    )


def attach_transformers_olmoe_runtime(
    model: Any,
    runtime: PagedExpertRuntime,
    *,
    implementation: str = "moevm_paged",
) -> None:
    """Attach a runtime to OLMoE layer modules, including meta-initialized models."""
    register_transformers_paged_experts(implementation)
    try:
        layers = model.model.layers
        config = model.config
    except AttributeError as exc:
        raise TypeError("expected an OlmoeForCausalLM-compatible model") from exc
    store = runtime.cache.store
    expected_layers = tuple(range(len(layers)))
    if store.layers != expected_layers:
        raise ValueError("model layer count does not match the expert store")
    if config.hidden_act != "silu":
        raise ValueError(f"paged OLMoE requires SiLU, got {config.hidden_act}")
    if (
        config.hidden_size != store.spec.hidden_size
        or config.intermediate_size != store.spec.intermediate_size
    ):
        raise ValueError("model dimensions do not match the expert store")
    for layer_index, layer_module in enumerate(layers):
        experts = layer_module.mlp.experts
        if experts.num_experts != len(store.experts_in_layer(layer_index)):
            raise ValueError(
                f"model expert count does not match store layer {layer_index}"
            )
        if experts.gate_up_proj.dtype != store.spec.dtype:
            raise ValueError(
                f"model expert dtype does not match store layer {layer_index}"
            )
    config._experts_implementation = implementation
    for layer_index, layer_module in enumerate(layers):
        experts = layer_module.mlp.experts
        experts._moevm_paged_runtime = runtime
        experts._moevm_layer_index = layer_index
        experts.config._experts_implementation = implementation


def load_non_expert_weights_into_meta_model(
    model: Any,
    store: SafetensorExpertStore,
    *,
    device: str | torch.device,
) -> tuple[str, ...]:
    """Materialize only non-expert parameters in an Accelerate meta model.

    Expert parameters intentionally remain on ``meta`` and are ignored by the
    registered paged forward.  Tensors are processed one at a time so peak host
    memory does not include a full state dict.
    """
    try:
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError("the non-expert meta loader requires accelerate") from exc

    loaded: list[str] = []
    for tensor_name, tensor in store.iter_non_expert_tensors():
        try:
            set_module_tensor_to_device(
                model,
                tensor_name,
                device,
                value=tensor,
                dtype=tensor.dtype,
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint tensor does not match the meta model: {tensor_name}"
            ) from exc
        loaded.append(tensor_name)
    return tuple(loaded)


def validate_transformers_paged_model(
    model: Any,
    store: SafetensorExpertStore,
    *,
    device: str | torch.device | None = None,
    runtime: PagedExpertRuntime | None = None,
) -> dict[str, int | str]:
    """Fail closed if a paged OLMoE model has unsafe meta or dtype state."""
    expert_parameter_pattern = re.compile(
        r"^model\.layers\.\d+\.mlp\.experts\."
        r"(gate_up_proj|down_proj)$"
    )
    expert_meta = 0
    non_expert_parameters = 0
    expected_device = torch.device(device) if device is not None else None
    if (
        expected_device is not None
        and expected_device.type == "cuda"
        and expected_device.index is None
    ):
        expected_device = torch.device("cuda", torch.cuda.current_device())
    for name, parameter in model.named_parameters():
        if expert_parameter_pattern.fullmatch(name):
            if parameter.device.type != "meta":
                raise RuntimeError(f"expert parameter was materialized eagerly: {name}")
            expert_meta += 1
            continue
        non_expert_parameters += 1
        if parameter.device.type == "meta":
            raise RuntimeError(f"non-expert parameter remains on meta: {name}")
        if expected_device is not None and parameter.device != expected_device:
            raise RuntimeError(
                f"non-expert parameter is on the wrong device for {name}: "
                f"{parameter.device} != {expected_device}"
            )
        if parameter.is_floating_point() and parameter.dtype != store.spec.dtype:
            raise RuntimeError(
                f"non-expert dtype mismatch for {name}: "
                f"{parameter.dtype} != {store.spec.dtype}"
            )

    meta_buffers = [
        name for name, buffer in model.named_buffers() if buffer.device.type == "meta"
    ]
    if meta_buffers:
        raise RuntimeError(f"model buffer remains on meta: {meta_buffers[0]}")
    expected_expert_parameters = 2 * len(store.layers)
    if expert_meta != expected_expert_parameters:
        raise RuntimeError(
            "unexpected paged expert parameter count: "
            f"{expert_meta} != {expected_expert_parameters}"
        )
    embedding = model.get_input_embeddings().weight
    if embedding.dtype != store.spec.dtype:
        raise RuntimeError(
            f"input embedding dtype mismatch: {embedding.dtype} != {store.spec.dtype}"
        )
    attached_runtime: PagedExpertRuntime | None = None
    implementation = getattr(model.config, "_experts_implementation", None)
    try:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError("paged model validation requires transformers") from exc
    if (
        not isinstance(implementation, str)
        or ALL_EXPERTS_FUNCTIONS.get(implementation)
        is not transformers_paged_experts_forward
    ):
        raise RuntimeError("model config does not resolve to the MoEVM paged backend")
    for layer_index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        candidate = getattr(experts, "_moevm_paged_runtime", None)
        attached_layer = getattr(experts, "_moevm_layer_index", None)
        if not isinstance(candidate, PagedExpertRuntime):
            raise RuntimeError(f"paged runtime is not attached to layer {layer_index}")
        if attached_layer != layer_index:
            raise RuntimeError(f"paged layer id mismatch at layer {layer_index}")
        if experts.config._experts_implementation != implementation:
            raise RuntimeError(f"expert backend mismatch at layer {layer_index}")
        if attached_runtime is None:
            attached_runtime = candidate
        elif attached_runtime is not candidate:
            raise RuntimeError("model layers do not share one paged runtime")
    if runtime is not None and attached_runtime is not runtime:
        raise RuntimeError("model is attached to an unexpected paged runtime")
    if attached_runtime is None or attached_runtime.cache.store is not store:
        raise RuntimeError("paged runtime is attached to a different expert store")
    if embedding.device != attached_runtime.cache.device:
        raise RuntimeError(
            "input embedding and expert cache must use the same target device"
        )
    return {
        "expert_meta_parameters": expert_meta,
        "non_expert_parameters": non_expert_parameters,
        "buffers": sum(1 for _ in model.named_buffers()),
        "cpu_buffers_allowed": sum(
            1 for _, buffer in model.named_buffers() if buffer.device.type == "cpu"
        ),
        "dtype": str(store.spec.dtype),
    }
