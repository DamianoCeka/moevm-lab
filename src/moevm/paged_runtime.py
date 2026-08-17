# Copyright 2026 Damiano Ceka
# Portions adapted and modified from Hugging Face Transformers:
# Copyright 2023 Mistral AI and the HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded, read-only expert paging primitives for supported MoE checkpoints.

This module deliberately keeps the first hardware runtime small and bounded:
only the experts selected by the router are read, staging memory is preallocated,
and every cache slot has a fixed allocation.  Synchronous execution remains the
default; async and adaptive paths are opt-in experiments.  It is an inference
prototype, not a training implementation.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Self

try:
    import torch
    import torch.nn.functional as torch_functional
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "moevm.paged_runtime requires the 'real-traces' optional dependencies"
    ) from exc

from .timeline_metrics import CudaInterval, summarize_cuda_timeline
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
    """Packed gated-MLP expert layout used by the supported Transformers models."""

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
            snapshot_path = supplied.resolve()
            index_path = snapshot_path / "model.safetensors.index.json"
        else:
            snapshot_path = supplied.parent.resolve()
            index_path = snapshot_path / supplied.name
        if not index_path.is_file():
            raise FileNotFoundError(f"safetensors index not found: {index_path}")

        # Hugging Face's canonical cache stores snapshot entries as symlinks to
        # content-addressed files in a sibling ``blobs`` directory. Preserve
        # the lexical snapshot directory instead of deriving it from the
        # resolved index symlink target.
        self.index_path = index_path
        self.snapshot_path = snapshot_path
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
            raise ValueError("index contains no supported per-expert weights")
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
                shard_path = self._resolve_shard_path(shard_name)
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
        return self._resolve_shard_path(shard_name)

    def _resolve_shard_path(self, shard_name: str) -> Path:
        posix_path = PurePosixPath(shard_name)
        windows_path = PureWindowsPath(shard_name)
        if (
            posix_path.is_absolute()
            or bool(windows_path.drive)
            or bool(windows_path.root)
            or "\\" in shard_name
            or any(part in ("", ".", "..") for part in posix_path.parts)
        ):
            raise ValueError(
                f"safetensors shard escapes snapshot directory: {shard_name}"
            )

        lexical_path = self.snapshot_path.joinpath(*posix_path.parts)
        resolved_parent = lexical_path.parent.resolve()
        if not resolved_parent.is_relative_to(self.snapshot_path):
            raise ValueError(
                f"safetensors shard escapes snapshot directory: {shard_name}"
            )
        resolved_path = lexical_path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"safetensors shard not found: {resolved_path}")
        return resolved_path

    def _validate_shard_paths(self) -> None:
        for shard_name in sorted(set(self._weight_map.values())):
            self._resolve_shard_path(shard_name)

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
    coalesced_requests: int = 0
    storage_loads: int = 0
    transfer_loads: int = 0
    admission_rejections: int = 0
    storage_failures: int = 0
    transfer_failures: int = 0
    pending_loads_peak: int = 0
    peak_staging_in_use: int = 0
    staging_waits: int = 0
    storage_queue_seconds: float = 0.0
    reader_queue_wait_seconds: float = 0.0
    staging_wait_seconds: float = 0.0
    proactive_h2d_slot_declines: int = 0
    demand_wait_seconds: float = 0.0
    adaptive_async_forwards: int = 0
    adaptive_sync_forwards: int = 0
    adaptive_async_experts: int = 0
    adaptive_sync_experts: int = 0

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
    coalesced_requests: int = 0
    storage_loads: int = 0
    transfer_loads: int = 0
    admission_rejections: int = 0
    storage_failures: int = 0
    transfer_failures: int = 0
    pending_loads_peak: int = 0
    peak_staging_in_use: int = 0
    staging_waits: int = 0
    storage_queue_seconds: float = 0.0
    reader_queue_wait_seconds: float = 0.0
    staging_wait_seconds: float = 0.0
    proactive_h2d_slot_declines: int = 0
    demand_wait_seconds: float = 0.0
    adaptive_async_forwards: int = 0
    adaptive_sync_forwards: int = 0
    adaptive_async_experts: int = 0
    adaptive_sync_experts: int = 0

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
            coalesced_requests=self.coalesced_requests,
            storage_loads=self.storage_loads,
            transfer_loads=self.transfer_loads,
            admission_rejections=self.admission_rejections,
            storage_failures=self.storage_failures,
            transfer_failures=self.transfer_failures,
            pending_loads_peak=self.pending_loads_peak,
            peak_staging_in_use=self.peak_staging_in_use,
            staging_waits=self.staging_waits,
            storage_queue_seconds=self.storage_queue_seconds,
            reader_queue_wait_seconds=self.reader_queue_wait_seconds,
            staging_wait_seconds=self.staging_wait_seconds,
            proactive_h2d_slot_declines=self.proactive_h2d_slot_declines,
            demand_wait_seconds=self.demand_wait_seconds,
            adaptive_async_forwards=self.adaptive_async_forwards,
            adaptive_sync_forwards=self.adaptive_sync_forwards,
            adaptive_async_experts=self.adaptive_async_experts,
            adaptive_sync_experts=self.adaptive_sync_experts,
        )


@dataclass(slots=True)
class _CudaTimelineEventSpan:
    """One CUDA-event interval emitted by a pass-scoped timeline capture."""

    lane: str
    key: ExpertKey
    sequence: int
    started_event: Any
    ended_event: Any


@dataclass(slots=True)
class _CudaTimelineTransferLedger:
    """One transfer a capture admitted before an I/O worker could see it."""

    # Keep a strong reference while the capture is open.  The runtime creates
    # short-lived ticket objects per layer, so an ``id(ticket)`` key alone can
    # be reused by Python for a later expert before model-call capture closes.
    ticket: Any
    key: ExpertKey
    state: str = "reserved"
    ready_event: Any | None = None


def _cuda_timeline_payload(
    origin_event: Any,
    spans: Iterable[_CudaTimelineEventSpan],
    *,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Materialize one same-device CUDA timeline after its events complete.

    The caller must synchronize the selected device before calling this helper.
    Keeping event-to-timestamp conversion separate from CUDA scheduling makes the
    evidence schema testable with small fake event objects on CPU-only hosts.
    """

    transfer_intervals: list[CudaInterval] = []
    compute_intervals: list[CudaInterval] = []
    raw_spans: list[dict[str, Any]] = []
    for span in spans:
        if not bool(span.started_event.query()) or not bool(span.ended_event.query()):
            raise RuntimeError("CUDA timeline event did not complete before capture")
        started_ms = float(origin_event.elapsed_time(span.started_event))
        ended_ms = float(origin_event.elapsed_time(span.ended_event))
        # CUDA events from both streams are causally ordered after the shared
        # origin.  A materially reversed pair would make the evidence invalid,
        # rather than something to silently clamp or reorder.
        if ended_ms < started_ms:
            raise RuntimeError("CUDA timeline event end precedes its start")
        name = f"{span.lane}:{span.sequence}:L{span.key.layer}:E{span.key.expert}"
        interval = CudaInterval(name=name, start_ms=started_ms, end_ms=ended_ms)
        target = transfer_intervals if span.lane == "h2d" else compute_intervals
        target.append(interval)
        raw_spans.append(
            {
                "lane": span.lane,
                "sequence": span.sequence,
                "layer": span.key.layer,
                "expert": span.key.expert,
                **interval.to_dict(),
            }
        )

    raw_spans.sort(
        key=lambda item: (
            float(item["start_ms"]),
            float(item["end_ms"]),
            str(item["lane"]),
            int(item["sequence"]),
        )
    )
    summary = summarize_cuda_timeline(
        transfers=transfer_intervals,
        compute=compute_intervals,
    )
    h2d_count = int(summary["transfer"]["interval_count"])
    compute_count = int(summary["compute"]["interval_count"])
    if h2d_count and compute_count:
        status = "measured"
        reason = None
    elif not h2d_count:
        status = "not_applicable"
        reason = "no expert H2D intervals were observed in this model call"
    else:
        status = "not_applicable"
        reason = "no expert compute intervals were observed in this model call"
    return {
        "schema_version": schema_version,
        "status": status,
        "method": "cuda_events_v1",
        "scope": "paged_expert_h2d_vs_expert_compute",
        "unit": "milliseconds",
        "complete": True,
        "reason": reason,
        "spans": raw_spans,
        "summary": {key: value for key, value in summary.items() if key != "intervals"},
    }


class _CudaTimelineCapture:
    """Collect H2D and paged-expert compute spans for one model invocation."""

    _ACTIVE = "active"
    _CLOSING = "closing"
    _FINISHED = "finished"
    _ABORTED = "aborted"

    def __init__(
        self,
        device: torch.device,
        *,
        transfer_loads_baseline: int | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA timeline capture requires a CUDA device")
        if transfer_loads_baseline is not None and transfer_loads_baseline < 0:
            raise ValueError("CUDA timeline transfer baseline cannot be negative")
        self.device = device
        self._condition = threading.Condition(threading.RLock())
        self._origin_event: Any | None = None
        self._spans: list[_CudaTimelineEventSpan] = []
        self._next_sequence = 0
        self._state = "new"
        # Every async ticket is admitted here before it becomes visible to the
        # I/O worker.  A worker must claim the entry before it records CUDA
        # work, then either commit its event pair or cancel it.  This closes
        # the old race where a worker could append an H2D span after the
        # forward thread had already finalized the capture.
        self._transfer_ledger: dict[int, _CudaTimelineTransferLedger] = {}
        # The cache credits async H2Ds only when a completion poll retires the
        # transfer.  A capture span is committed at enqueue time, so compare
        # against a cache metric delta scoped by snapshots around this capture.
        self._transfer_loads_baseline = transfer_loads_baseline
        self._transfer_coverage: dict[str, int] | None = None
        self._incomplete_reason: str | None = None
        self.result: dict[str, Any] | None = None

    @property
    def _schema_version(self) -> int:
        """Keep unscoped low-level events readable as legacy v1 evidence."""

        return 2 if self._transfer_loads_baseline is not None else 1

    def begin(self, compute_stream: Any) -> None:
        with self._condition:
            if self._state != "new":
                raise RuntimeError("CUDA timeline capture has already started")
            origin = torch.cuda.Event(enable_timing=True)
            origin.record(compute_stream)
            self._origin_event = origin
            self._state = self._ACTIVE

    @property
    def origin_event(self) -> Any:
        with self._condition:
            if self._origin_event is None:
                raise RuntimeError("CUDA timeline capture has not started")
            return self._origin_event

    def wait_on_origin(self, stream: Any) -> None:
        """Make an instrumented stream causally comparable with the origin."""
        stream.wait_event(self.origin_event)

    def reserve_transfer(self, ticket: ExpertLoadTicket) -> bool:
        """Admit one async ticket before the I/O worker can dequeue it.

        The cache calls this while the ticket is still private to the submitter.
        Returning ``False`` means the capture is already closing, so the cache
        must keep executing normally but must not attribute later CUDA work to
        this trace.
        """

        token = id(ticket)
        with self._condition:
            if self._state != self._ACTIVE:
                self._mark_incomplete_locked(
                    "an async ticket was admitted after CUDA timeline close began"
                )
                return False
            existing = self._transfer_ledger.get(token)
            if existing is not None:
                if existing.key != ticket.key:  # pragma: no cover - identity invariant
                    self._mark_incomplete_locked(
                        "a CUDA timeline ticket identity was reused for another expert"
                    )
                    return False
                return True
            self._transfer_ledger[token] = _CudaTimelineTransferLedger(
                ticket=ticket,
                key=ticket.key,
            )
            return True

    def claim_transfer(self, ticket: ExpertLoadTicket) -> bool:
        """Give one scheduler authority to enqueue this ticket's H2D.

        The claim is deliberately taken before ``wait_on_origin`` and timing
        events.  ``begin_close`` rejects unclaimed work, while a claim already
        in progress is allowed to commit and is waited before finalization.
        """

        token = id(ticket)
        with self._condition:
            entry = self._transfer_ledger.get(token)
            if self._state != self._ACTIVE or entry is None:
                if self._state == self._ACTIVE:
                    self._mark_incomplete_locked(
                        "an async H2D was not reserved by the CUDA timeline"
                    )
                return False
            if entry.state != "reserved":
                self._mark_incomplete_locked(
                    "a CUDA timeline transfer was claimed more than once"
                )
                return False
            entry.state = "claimed"
            self._condition.notify_all()
            return True

    def commit_transfer(
        self,
        ticket: ExpertLoadTicket,
        started_event: Any,
        ended_event: Any,
    ) -> bool:
        """Commit an already-claimed H2D event pair to the trace ledger."""

        token = id(ticket)
        with self._condition:
            entry = self._transfer_ledger.get(token)
            if (
                entry is None
                or entry.state != "claimed"
                or self._state not in (self._ACTIVE, self._CLOSING)
            ):
                self._mark_incomplete_locked(
                    "a CUDA timeline H2D committed outside its capture lifetime"
                )
                return False
            self._record_locked("h2d", ticket.key, started_event, ended_event)
            # Keep the ready event in the ledger until the close path has
            # synchronized the device.  A recorded span alone proves only that
            # work was enqueued, not that its endpoint is safe to timestamp.
            entry.state = "enqueued"
            entry.ready_event = ended_event
            self._condition.notify_all()
            return True

    def cancel_transfer(self, ticket: ExpertLoadTicket, reason: str) -> None:
        """Failure-close the capture when an admitted H2D cannot be measured."""

        token = id(ticket)
        with self._condition:
            if self._transfer_ledger.pop(token, None) is not None:
                self._mark_incomplete_locked(reason)
                self._condition.notify_all()

    def invalidate(self, reason: str) -> None:
        """Mark this evidence unusable without changing runtime behavior."""

        with self._condition:
            self._mark_incomplete_locked(reason)
            self._condition.notify_all()

    def record_transfer(
        self, key: ExpertKey, started_event: Any, ended_event: Any
    ) -> bool:
        """Record a synchronous H2D that has no worker ticket ledger."""

        with self._condition:
            if self._state != self._ACTIVE:
                self._mark_incomplete_locked(
                    "a CUDA timeline H2D was recorded after capture close began"
                )
                return False
            self._record_locked("h2d", key, started_event, ended_event)
            return True

    def begin_compute(self, key: ExpertKey, stream: Any) -> Any:
        # A caller may enter the capture on one stream and run a model forward
        # on another.  Make every instrumented compute stream depend on the
        # common origin before using elapsed-time comparisons across streams.
        with self._condition:
            if self._state != self._ACTIVE:
                raise RuntimeError("cannot start compute after CUDA timeline close")
            self.wait_on_origin(stream)
            started = torch.cuda.Event(enable_timing=True)
            started.record(stream)
            return started

    def end_compute(self, key: ExpertKey, started_event: Any, stream: Any) -> None:
        ended = torch.cuda.Event(enable_timing=True)
        ended.record(stream)
        with self._condition:
            if self._state != self._ACTIVE:
                self._mark_incomplete_locked(
                    "a CUDA timeline compute span ended after capture close began"
                )
                return
            self._record_locked("expert_compute", key, started_event, ended)

    def _record_locked(
        self,
        lane: str,
        key: ExpertKey,
        started_event: Any,
        ended_event: Any,
    ) -> None:
        self._spans.append(
            _CudaTimelineEventSpan(
                lane=lane,
                key=key,
                sequence=self._next_sequence,
                started_event=started_event,
                ended_event=ended_event,
            )
        )
        self._next_sequence += 1

    def _mark_incomplete_locked(self, reason: str) -> None:
        if self._incomplete_reason is None:
            self._incomplete_reason = reason
        if self._state == self._FINISHED:
            # A late CUDA readiness failure can be observed by the cache
            # immediately after scope snapshot.  Do not leave a stale
            # ``complete`` result visible in that race.
            self.result = self._incomplete_payload_locked()

    def _begin_close(self) -> None:
        """Stop admission and cancel worker work that never claimed H2D."""

        with self._condition:
            if self._state == self._ACTIVE:
                self._state = self._CLOSING
                unclaimed = tuple(
                    token
                    for token, entry in self._transfer_ledger.items()
                    if entry.state == "reserved"
                )
                if unclaimed:
                    for token in unclaimed:
                        del self._transfer_ledger[token]
                    self._mark_incomplete_locked(
                        "an async H2D was cancelled before it could be measured"
                    )
                self._condition.notify_all()
                return
            if self._state in (self._CLOSING, self._FINISHED, self._ABORTED):
                return
            raise RuntimeError("CUDA timeline capture has not started")

    def _wait_for_claimed_transfers(self) -> None:
        """Wait until all pre-close worker claims commit or cancel."""

        with self._condition:
            while any(
                entry.state == "claimed" for entry in self._transfer_ledger.values()
            ):
                self._condition.wait(timeout=0.002)

    def _settle_enqueued_transfers(self) -> None:
        """Retire H2Ds only after device synchronization made events complete."""

        with self._condition:
            for token, entry in tuple(self._transfer_ledger.items()):
                if entry.state != "enqueued":
                    self._mark_incomplete_locked(
                        "an async H2D did not settle before CUDA timeline snapshot"
                    )
                    del self._transfer_ledger[token]
                    continue
                ready = entry.ready_event
                try:
                    complete = ready is not None and bool(ready.query())
                except Exception:  # noqa: BLE001 - evidence must fail closed
                    complete = False
                if not complete:
                    self._mark_incomplete_locked(
                        "an async H2D event was incomplete after CUDA synchronization"
                    )
                del self._transfer_ledger[token]
            self._condition.notify_all()

    def _reconcile_transfer_coverage(self, transfer_loads_after: int) -> None:
        """Fail closed unless cache retirement matches committed H2D spans."""

        with self._condition:
            baseline = self._transfer_loads_baseline
            if baseline is None:
                return
            transfer_loads_delta = transfer_loads_after - baseline
            h2d_span_count = sum(span.lane == "h2d" for span in self._spans)
            self._transfer_coverage = {
                "cache_transfer_loads_delta": transfer_loads_delta,
                "h2d_span_count": h2d_span_count,
            }
            if transfer_loads_delta < 0:
                self._mark_incomplete_locked(
                    "cache transfer metric decreased during CUDA timeline capture"
                )
            elif h2d_span_count != transfer_loads_delta:
                self._mark_incomplete_locked(
                    "CUDA timeline H2D coverage mismatch: "
                    f"{h2d_span_count} spans but cache recorded "
                    f"{transfer_loads_delta} transfers"
                )

    def _incomplete_payload_locked(self) -> dict[str, Any]:
        summary = summarize_cuda_timeline(transfers=(), compute=())
        payload: dict[str, Any] = {
            "schema_version": self._schema_version,
            "status": "incomplete",
            "method": "cuda_events_v1",
            "scope": "paged_expert_h2d_vs_expert_compute",
            "unit": "milliseconds",
            "complete": False,
            "reason": self._incomplete_reason
            or "CUDA timeline capture did not settle cleanly",
            "spans": [],
            "summary": {
                key: value for key, value in summary.items() if key != "intervals"
            },
        }
        if self._transfer_coverage is not None:
            payload["coverage"] = dict(self._transfer_coverage)
        return payload

    def finish(
        self,
        *,
        transfer_loads_after_synchronize: Callable[[], int] | None = None,
    ) -> dict[str, Any]:
        self._begin_close()
        self._wait_for_claimed_transfers()
        try:
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - evidence must fail closed
            self.invalidate(f"CUDA timeline synchronization failed: {exc}")
        self._settle_enqueued_transfers()
        with self._condition:
            needs_coverage = (
                self._transfer_loads_baseline is not None
                and self._state not in (self._FINISHED, self._ABORTED)
                and self._incomplete_reason is None
            )
        if needs_coverage:
            if transfer_loads_after_synchronize is None:
                self.invalidate(
                    "CUDA timeline cache transfer metric snapshot was not provided"
                )
            else:
                try:
                    transfer_loads_after = transfer_loads_after_synchronize()
                except Exception as exc:  # noqa: BLE001 - evidence must fail closed
                    self.invalidate(
                        f"CUDA timeline cache transfer metric snapshot failed: {exc}"
                    )
                else:
                    self._reconcile_transfer_coverage(transfer_loads_after)
        with self._condition:
            if self._state in (self._FINISHED, self._ABORTED):
                if self.result is None:  # pragma: no cover - internal invariant
                    raise RuntimeError("finished CUDA timeline is missing its result")
                return self.result
            if self._incomplete_reason is not None:
                self.result = self._incomplete_payload_locked()
                self._state = self._FINISHED
                return self.result
            origin = self.origin_event
            spans = tuple(self._spans)
        try:
            result = _cuda_timeline_payload(
                origin,
                spans,
                schema_version=self._schema_version,
            )
        except Exception as exc:  # noqa: BLE001 - evidence must fail closed
            self.invalidate(f"CUDA timeline materialization failed: {exc}")
            with self._condition:
                self.result = self._incomplete_payload_locked()
                self._state = self._FINISHED
                return self.result
        with self._condition:
            # No new claims are possible after _begin_close(), so this is the
            # final immutable trace visible to callers after scope exit.
            if self._incomplete_reason is not None:
                self.result = self._incomplete_payload_locked()
            else:
                if self._transfer_coverage is not None:
                    result["coverage"] = dict(self._transfer_coverage)
                self.result = result
            self._state = self._FINISHED
            return self.result

    def abort(self) -> None:
        """Release event references after a failed model invocation."""
        with self._condition:
            if self._state in (self._FINISHED, self._ABORTED):
                return
            self._state = self._ABORTED
            self._transfer_ledger.clear()
            self._mark_incomplete_locked("the model invocation failed during capture")
            self._spans.clear()
            self.result = self._incomplete_payload_locked()
            self._condition.notify_all()


class _CudaTimelineCaptureScope:
    """Context manager that binds one capture to a paged runtime invocation."""

    def __init__(self, runtime: PagedExpertRuntime) -> None:
        self._runtime = runtime
        self._capture: _CudaTimelineCapture | None = None
        self._holds_forward_lock = False

    def __enter__(self) -> _CudaTimelineCapture:
        # Keep one runtime exclusively owned for the whole model invocation.
        # `runtime.forward()` is reentrant on this RLock for the owner thread,
        # while another thread cannot append unrelated spans to this capture.
        self._runtime._forward_lock.acquire()
        self._holds_forward_lock = True
        try:
            self._capture = self._runtime._begin_cuda_timeline_capture()
            return self._capture
        except BaseException:
            self._runtime._forward_lock.release()
            self._holds_forward_lock = False
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> bool:
        capture = self._capture
        if capture is None:  # pragma: no cover - context protocol invariant
            return False
        try:
            self._runtime._end_cuda_timeline_capture(
                capture,
                failed=exc_type is not None,
            )
        finally:
            if self._holds_forward_lock:
                self._runtime._forward_lock.release()
                self._holds_forward_lock = False
        return False


@dataclass(slots=True)
class ExpertLoadTicket:
    """Single-flight handle for one asynchronous expert load."""

    key: ExpertKey
    queued_at: float
    request_clock: int
    reader_queue_enqueued_at: float | None = None
    storage_done: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)
    stage_index: int | None = None
    destination_slot: int | None = None
    destination_generation: int = 0
    bytes_read: int = 0
    error: BaseException | None = None
    ready_event: Any | None = None
    transfer_started_event: Any | None = None
    state: str = "queued"
    counted_as_hit: bool = False
    cuda_timeline: _CudaTimelineCapture | None = None
    # Internal forward lookaheads may be copied into an otherwise empty device
    # slot before the router reaches them.  A reservation is deliberately
    # physical-only: it must never be published in the logical cache maps
    # until the corresponding demand is accounted for.
    is_lookahead: bool = False
    demanded: bool = False
    reservation_active: bool = False
    discard_requested: bool = False
    transfer_enqueued: threading.Event = field(default_factory=threading.Event)


@dataclass(slots=True)
class _ExpertLease:
    cache: ExpertSlotCache
    slot: int
    generation: int
    weights: ExpertWeights
    released: bool = False

    def release_after(self, stream: Any | None = None) -> None:
        if self.released:
            return
        self.cache._release_lease(self.slot, self.generation, stream)
        self.released = True


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
        pipeline_mode: str = "sync",
    ) -> None:
        if (capacity is None) == (capacity_per_layer is None):
            raise ValueError("set exactly one of capacity or capacity_per_layer")
        selected_capacity = capacity if capacity is not None else capacity_per_layer
        if selected_capacity is None or selected_capacity <= 0:
            raise ValueError("capacity must be positive")
        if staging_slots <= 0:
            raise ValueError("staging_slots must be positive")
        if pipeline_mode not in ("sync", "async", "adaptive"):
            raise ValueError("pipeline_mode must be 'sync', 'async', or 'adaptive'")
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
        self.pipeline_mode = pipeline_mode
        self._async_infrastructure_enabled = pipeline_mode in ("async", "adaptive")
        if (
            self._uses_async_infrastructure
            and self.device.type == "cuda"
            and not self.pin_staging
        ):
            raise ValueError("the CUDA async-capable pipeline requires pinned staging")
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
        self._condition = threading.Condition(self._lock)
        self.execution_lock = threading.RLock()
        self._slot_generation: list[int] = [0] * self.capacity
        self._slot_copying: list[bool] = [False] * self.capacity
        self._slot_pin_count: list[int] = [0] * self.capacity
        self._slot_last_use_event: list[Any | None] = [None] * self.capacity
        # A failed H2D may be reported after a forward thread has already
        # leased the destination slot and queued compute behind its ready
        # event.  Such a slot cannot be returned to either allocator until the
        # lease releases and gives us an ordering event for that compute.
        self._slot_quarantined: list[bool] = [False] * self.capacity
        # A physical-only reservation owns a device slot while an internal
        # lookahead H2D is in flight or has completed ahead of demand.  It is
        # intentionally distinct from _slot_to_key/_key_to_slot so that
        # requests, misses, LRU timestamps, and evictions stay demand-driven.
        self._slot_reservation: list[ExpertLoadTicket | None] = [None] * self.capacity
        self._stage_owner: list[ExpertLoadTicket | None] = [None] * staging_slots
        self._stage_state: list[str] = ["free"] * staging_slots
        self._pending_by_key: dict[ExpertKey, ExpertLoadTicket] = {}
        self._unobserved_errors: dict[int, Exception] = {}
        self._jobs: queue.Queue[ExpertLoadTicket | None] | None = None
        # Tracks jobs accepted by the queue until the I/O worker acknowledges
        # them with task_done().  `wait_idle()` includes it in its drain gate,
        # so an abandoned queued lookahead cannot outlive a mode switch.
        self._queued_job_count = 0
        self._worker: threading.Thread | None = None
        self._accepting = True
        self._closed = False
        self._close_lock = threading.Lock()
        self._transfer_stream: Any | None = None
        # Both the foreground and the persistent I/O worker can submit work to
        # this one CUDA stream.  Serialize a complete dependency/start/copy/end
        # sequence so timing spans cannot nest or interleave on the same lane.
        self._transfer_submit_lock = threading.Lock()
        self._pipeline_error: BaseException | None = None
        if self._uses_async_infrastructure:
            self._jobs = queue.Queue(maxsize=staging_slots)
            if self.device.type == "cuda":
                self._transfer_stream = torch.cuda.Stream(device=self.device)

    @property
    def _uses_async_infrastructure(self) -> bool:
        return self._async_infrastructure_enabled

    def set_pipeline_mode(self, pipeline_mode: str) -> None:
        """Switch the active data path at a drained pass boundary.

        A cache constructed as async-capable may switch between ``sync``,
        ``async``, and ``adaptive`` without reallocating its bounded buffers.
        A cache constructed in synchronous mode cannot enable asynchronous
        execution later because it has no worker/stream lifecycle to own.
        """
        if pipeline_mode not in ("sync", "async", "adaptive"):
            raise ValueError("pipeline_mode must be 'sync', 'async', or 'adaptive'")
        with self.execution_lock:
            if self._closed:
                raise RuntimeError("expert slot cache is closed")
            if pipeline_mode in ("async", "adaptive") and not (
                self._async_infrastructure_enabled
            ):
                raise RuntimeError(
                    "cache was not constructed with async infrastructure"
                )
            self.wait_idle()
            self.pipeline_mode = pipeline_mode

    @property
    def allocated_cache_bytes(self) -> int:
        return self.capacity * self.store.spec.size_bytes

    @property
    def allocated_staging_bytes(self) -> int:
        return self.staging_slots * self.store.spec.size_bytes

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("expert slot cache is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_worker(self) -> None:
        if not self._uses_async_infrastructure:
            return
        with self._condition:
            if not self._accepting or self._closed:
                raise RuntimeError("expert slot cache is closed")
            if self._pipeline_error is not None:
                raise RuntimeError("asynchronous expert pipeline failed") from (
                    self._pipeline_error
                )
            if self._worker is not None:
                return
            worker = threading.Thread(
                target=self._io_worker,
                name="moevm-expert-io",
                daemon=False,
            )
            self._worker = worker
            worker.start()

    def _claim_staging_slot(self, ticket: ExpertLoadTicket) -> int:
        waited = False
        staging_wait_started: float | None = None
        while True:
            with self._condition:
                for stage_index, state in enumerate(self._stage_state):
                    if state == "free":
                        self._stage_state[stage_index] = "reading"
                        self._stage_owner[stage_index] = ticket
                        ticket.stage_index = stage_index
                        in_use = sum(state != "free" for state in self._stage_state)
                        self._metrics.peak_staging_in_use = max(
                            self._metrics.peak_staging_in_use,
                            in_use,
                        )
                        if staging_wait_started is not None:
                            self._metrics.staging_wait_seconds += (
                                time.perf_counter() - staging_wait_started
                            )
                        return stage_index
                if not waited:
                    self._metrics.staging_waits += 1
                    waited = True
                    staging_wait_started = time.perf_counter()
            # The worker may be the only thread alive while an H2D owns every
            # staging buffer.  Poll outside the condition so completed
            # speculative copies release their buffers without a blocking
            # dispatcher or a second worker thread.
            self._poll_transfer_completions()
            with self._condition:
                self._condition.wait(timeout=0.005)

    def _free_staging_slot_locked(
        self,
        stage_index: int,
        ticket: ExpertLoadTicket,
    ) -> None:
        if self._stage_owner[stage_index] is not ticket:
            return
        self._stage_owner[stage_index] = None
        self._stage_state[stage_index] = "free"
        self._condition.notify_all()

    def _retire_undemanded_lookahead_locked(
        self,
        ticket: ExpertLoadTicket,
        *,
        worker_owns_job: bool = False,
    ) -> bool:
        """Discard completed speculative work without changing logical cache state.

        A read-only lookahead may be abandoned at a drained boundary or while
        closing.  Reads already in progress cannot be cancelled safely, and a
        CUDA H2D must retain staging ownership until its completion event fires;
        callers therefore retry after marking ``discard_requested``.  Returning
        ``False`` means the worker or completion poll still owns that progress.
        """

        if not ticket.is_lookahead or ticket.demanded:
            return False
        ticket.discard_requested = True
        if ticket.state == "prefetching":
            return False
        if ticket.state in ("queued", "reading") and not worker_owns_job:
            # The worker may already have dequeued this ticket or may do so
            # next.  Keep it pending until the owning worker acknowledges the
            # cancellation, rather than letting a drained boundary return with
            # unseen queue work still able to claim a staging slot.
            return False
        if ticket.state not in ("queued", "reading", "ready", "prefetched"):
            return ticket.state == "discarded"

        if ticket.state in ("queued", "reading", "ready"):
            stage_index = ticket.stage_index
            if stage_index is not None:
                self._free_staging_slot_locked(stage_index, ticket)
        elif ticket.state == "prefetched":
            slot = ticket.destination_slot
            if slot is None:  # pragma: no cover - reservation invariant
                raise RuntimeError("prefetched lookahead has no reserved slot")
            if self._slot_reservation[slot] is not ticket:
                raise RuntimeError("prefetched lookahead lost its reservation")
            if self._slot_pin_count[slot] != 0:
                # Promotion normally clears the reservation before a lease is
                # pinned.  Preserve that safety property even if a future
                # transition temporarily exposes both states.
                return False
            # H2D completed before the ticket reached ``prefetched``.  Its
            # staging slot has already been released, so this only returns an
            # otherwise-empty device slot to the physical pool.
            self._slot_reservation[slot] = None
            self._slot_copying[slot] = False
            self._last_used[slot] = 0
            self._slot_last_use_event[slot] = None
            ticket.reservation_active = False

        if self._pending_by_key.get(ticket.key) is ticket:
            del self._pending_by_key[ticket.key]
        if ticket.cuda_timeline is not None:
            ticket.cuda_timeline.cancel_transfer(
                ticket,
                "an async lookahead was cancelled before capture completion",
            )
        ticket.state = "discarded"
        ticket.storage_done.set()
        ticket.completed.set()
        ticket.transfer_enqueued.set()
        self._condition.notify_all()
        return True

    def _select_proactive_slot_locked(self, key: ExpertKey) -> int | None:
        """Return an empty physical slot suitable for a speculative H2D.

        This deliberately never chooses a resident victim.  A proactive copy
        can consume idle capacity, but is not allowed to evict an expert before
        the router actually asks for the lookahead key.
        """

        empty = tuple(
            slot
            for slot in self._async_candidate_slots(key)
            if self._slot_to_key[slot] is None
            and self._slot_reservation[slot] is None
            and not self._slot_copying[slot]
            and not self._slot_quarantined[slot]
            and self._slot_pin_count[slot] == 0
        )
        return min(empty) if empty else None

    def _schedule_proactive_lookahead_transfer(
        self,
        ticket: ExpertLoadTicket,
    ) -> bool:
        """Non-blockingly enqueue one physical-only H2D from the I/O worker.

        A capture ticket is reserved before the worker can dequeue it.  The
        worker claims that reservation before it makes the transfer stream wait
        on the common origin or records timing events; closing captures reject
        new claims but leave the normal, uninstrumented cache path intact.
        """

        if self.device.type != "cuda":
            return False
        with self._condition:
            if (
                not self._accepting
                or self._closed
                or not ticket.is_lookahead
                or ticket.demanded
                or ticket.discard_requested
                or ticket.state != "ready"
            ):
                return False
            stage_index = ticket.stage_index
            if stage_index is None:  # pragma: no cover - storage invariant
                raise RuntimeError("ready lookahead has no staging slot")
            slot = self._select_proactive_slot_locked(ticket.key)
            if slot is None:
                self._metrics.proactive_h2d_slot_declines += 1
                return False
            wait_events: list[Any] = []
            last_use = self._slot_last_use_event[slot]
            if last_use is not None:
                wait_events.append(last_use)
            self._slot_generation[slot] += 1
            ticket.destination_slot = slot
            ticket.destination_generation = self._slot_generation[slot]
            ticket.reservation_active = True
            ticket.transfer_enqueued.clear()
            self._slot_reservation[slot] = ticket
            self._slot_copying[slot] = True
            self._stage_state[stage_index] = "h2d"
            ticket.state = "prefetching"
            staging_gate_up = self._staging_gate_up[stage_index]
            staging_down = self._staging_down[stage_index]

        cuda_timeline = ticket.cuda_timeline
        timeline_claimed = False
        try:
            # CUDA device selection is thread-local.  The transfer stream is
            # created by the cache owner but can safely be submitted from this
            # persistent I/O worker once its device is selected here.
            torch.cuda.set_device(self.device)
            if cuda_timeline is not None:
                timeline_claimed = cuda_timeline.claim_transfer(ticket)
                if timeline_claimed:
                    if self._transfer_stream is None:  # pragma: no cover - invariant
                        raise RuntimeError("CUDA transfer stream is unavailable")
                    # Submit the origin dependency under the same transfer
                    # lock as timing/copy events, so it is immediately before
                    # this H2D's start event on the shared lane.
                    wait_events.append(cuda_timeline.origin_event)
            started_event, ready_event = self._enqueue_staging_to_slot(
                slot,
                staging_gate_up,
                staging_down,
                tuple(wait_events),
            )
            with self._condition:
                ticket.transfer_started_event = started_event
                ticket.ready_event = ready_event
                ticket.transfer_enqueued.set()
                self._condition.notify_all()
            if timeline_claimed:
                cuda_timeline.commit_transfer(ticket, started_event, ready_event)
            return True
        except Exception as exc:  # noqa: BLE001 - CUDA worker boundary
            if timeline_claimed:
                cuda_timeline.cancel_transfer(
                    ticket,
                    f"worker H2D enqueue failed: {exc}",
                )
            self._fail_transfer(ticket, exc)
            return False

    def _io_worker(self) -> None:
        jobs = self._jobs
        if jobs is None:  # pragma: no cover - constructor invariant
            return
        while True:
            ticket = jobs.get()
            try:
                if ticket is None:
                    return
                dequeued_at = time.perf_counter()
                with self._condition:
                    reader_queue_enqueued_at = ticket.reader_queue_enqueued_at
                    if reader_queue_enqueued_at is not None:
                        self._metrics.reader_queue_wait_seconds += (
                            dequeued_at - reader_queue_enqueued_at
                        )
                        ticket.reader_queue_enqueued_at = None
                    if ticket.discard_requested and not ticket.demanded:
                        self._retire_undemanded_lookahead_locked(
                            ticket,
                            worker_owns_job=True,
                        )
                        continue
                stage_index = self._claim_staging_slot(ticket)
                started = time.perf_counter()
                with self._condition:
                    # A cancellation can arrive while this worker waits for a
                    # staging slot.  Re-check after ownership is established;
                    # the worker, not wait_idle(), acknowledges that job.
                    if ticket.discard_requested and not ticket.demanded:
                        self._retire_undemanded_lookahead_locked(
                            ticket,
                            worker_owns_job=True,
                        )
                        continue
                    self._metrics.storage_queue_seconds += started - ticket.queued_at
                    ticket.state = "reading"
                try:
                    bytes_read = self.store.load_into(
                        ticket.key,
                        self._staging_gate_up[stage_index],
                        self._staging_down[stage_index],
                    )
                except Exception as exc:  # noqa: BLE001 - worker transports failures
                    if ticket.cuda_timeline is not None:
                        ticket.cuda_timeline.cancel_transfer(
                            ticket,
                            f"worker expert read failed: {exc}",
                        )
                    elapsed = time.perf_counter() - started
                    with self._condition:
                        ticket.error = exc
                        ticket.state = "failed"
                        self._unobserved_errors[id(ticket)] = exc
                        self._metrics.storage_seconds += elapsed
                        self._metrics.storage_failures += 1
                        if self._pending_by_key.get(ticket.key) is ticket:
                            del self._pending_by_key[ticket.key]
                        self._free_staging_slot_locked(stage_index, ticket)
                        ticket.storage_done.set()
                        ticket.completed.set()
                        ticket.transfer_enqueued.set()
                        self._condition.notify_all()
                    continue

                elapsed = time.perf_counter() - started
                with self._condition:
                    ticket.bytes_read = bytes_read
                    ticket.state = "ready"
                    self._stage_state[stage_index] = "ready"
                    self._metrics.storage_seconds += elapsed
                    self._metrics.storage_bytes += bytes_read
                    self._metrics.storage_loads += 1
                    self._condition.notify_all()
                    if ticket.discard_requested and not ticket.demanded:
                        self._retire_undemanded_lookahead_locked(
                            ticket,
                            worker_owns_job=True,
                        )
                        continue

                # Only internal lookaheads are eligible.  This call never
                # waits for a slot and never evicts a logical resident.
                self._schedule_proactive_lookahead_transfer(ticket)
                with self._condition:
                    ticket.storage_done.set()
                    self._condition.notify_all()
            finally:
                jobs.task_done()
                if ticket is not None:
                    with self._condition:
                        self._queued_job_count -= 1
                        if self._queued_job_count < 0:  # pragma: no cover
                            raise RuntimeError("asynchronous job count underflow")
                        self._condition.notify_all()

    def _new_resident_ticket_locked(
        self,
        key: ExpertKey,
        slot: int,
        *,
        account_request: bool,
        is_lookahead: bool = False,
    ) -> ExpertLoadTicket:
        # A resident hit has no H2D to reserve at submit time.  Preserve its
        # lookahead role, but do not retain a capture that may finish before a
        # later eviction turns this placeholder into a real queued transfer.
        ticket = ExpertLoadTicket(
            key=key,
            queued_at=time.perf_counter(),
            request_clock=self._clock if account_request else 0,
            destination_slot=slot,
            destination_generation=self._slot_generation[slot],
            state="resident",
            counted_as_hit=account_request,
            is_lookahead=is_lookahead,
            demanded=account_request,
        )
        ticket.storage_done.set()
        ticket.completed.set()
        return ticket

    def submit(self, key: ExpertKey) -> ExpertLoadTicket:
        """Submit one bounded, coalesced storage load in async mode."""
        if self.pipeline_mode != "async":
            raise RuntimeError("submit requires pipeline_mode='async'")
        with self.execution_lock:
            return self._submit_locked(key, account_request=True)

    def _submit_lookahead(
        self,
        key: ExpertKey,
        *,
        cuda_timeline: _CudaTimelineCapture | None = None,
    ) -> ExpertLoadTicket:
        """Schedule storage without changing demand or LRU accounting."""
        with self.execution_lock:
            return self._submit_locked(
                key,
                account_request=False,
                cuda_timeline=cuda_timeline,
            )

    def _submit_locked(
        self,
        key: ExpertKey,
        *,
        account_request: bool,
        cuda_timeline: _CudaTimelineCapture | None = None,
    ) -> ExpertLoadTicket:
        if not self._uses_async_infrastructure:
            raise RuntimeError("submit requires an async-capable pipeline mode")
        if key not in self.store:
            raise KeyError(f"unknown expert: {key.compact()}")
        self._ensure_worker()
        with self._condition:
            if not self._accepting or self._closed:
                raise RuntimeError("expert slot cache is closed")
            if self._pipeline_error is not None:
                raise RuntimeError("asynchronous expert pipeline failed") from (
                    self._pipeline_error
                )
            if account_request:
                self._clock += 1
                self._metrics.requests += 1
            existing_slot = self._key_to_slot.get(key)
            if existing_slot is not None:
                if account_request:
                    self._last_used[existing_slot] = self._clock
                    self._metrics.hits += 1
                return self._new_resident_ticket_locked(
                    key,
                    existing_slot,
                    account_request=account_request,
                    is_lookahead=not account_request,
                )
            existing_ticket = self._pending_by_key.get(key)
            if existing_ticket is not None:
                if account_request:
                    existing_ticket.request_clock = self._clock
                    existing_ticket.demanded = True
                    self._metrics.misses += 1
                    self._metrics.coalesced_requests += 1
                    self._admit_proactive_reservation_locked(existing_ticket)
                return existing_ticket
            if account_request:
                self._metrics.misses += 1
            ticket = ExpertLoadTicket(
                key=key,
                queued_at=time.perf_counter(),
                request_clock=self._clock if account_request else 0,
                cuda_timeline=cuda_timeline,
                is_lookahead=not account_request,
                demanded=account_request,
            )
            self._pending_by_key[key] = ticket
            self._metrics.pending_loads_peak = max(
                self._metrics.pending_loads_peak,
                len(self._pending_by_key),
            )

        self._enqueue_ticket_after_timeline_reservation(
            ticket,
            rollback_request=account_request,
        )
        return ticket

    def _enqueue_ticket_after_timeline_reservation(
        self,
        ticket: ExpertLoadTicket,
        *,
        rollback_request: bool = False,
    ) -> None:
        """Make a queued ticket visible only after its timeline admission.

        A cache-hit placeholder can become a real load after its resident slot
        is evicted.  Both that stale-resident path and a reclaimed lookahead
        re-enter the I/O queue here, so they need the same reserve-before-queue
        rule as a newly submitted ticket.
        """

        # Register the ticket while it is still private to this submitter.
        # `_enqueue_ticket()` is the first point at which the persistent I/O
        # worker can observe it, so moving this below the queue put would let a
        # worker append an unowned H2D span during capture close.
        cuda_timeline = ticket.cuda_timeline
        if cuda_timeline is not None and not cuda_timeline.reserve_transfer(ticket):
            # Instrumentation must not make the actual cache request fail.
            # The capture is already marked incomplete; detach the ticket so a
            # later normal transfer cannot try to record into a closed scope.
            ticket.cuda_timeline = None
            cuda_timeline = None
        try:
            self._enqueue_ticket(ticket, rollback_request=rollback_request)
        except Exception:
            if cuda_timeline is not None:
                cuda_timeline.cancel_transfer(
                    ticket,
                    "an async ticket could not enter the I/O worker queue",
                )
            raise

    def _enqueue_ticket(
        self,
        ticket: ExpertLoadTicket,
        *,
        rollback_request: bool = False,
    ) -> None:
        jobs = self._jobs
        if jobs is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("asynchronous job queue is unavailable")
        with self._condition:
            self._queued_job_count += 1
        try:
            ticket.reader_queue_enqueued_at = time.perf_counter()
            jobs.put_nowait(ticket)
        except queue.Full as exc:
            error = RuntimeError(
                "asynchronous expert queue is full; resolve submitted work first"
            )
            with self._condition:
                self._queued_job_count -= 1
                ticket.reader_queue_enqueued_at = None
                if self._pending_by_key.get(ticket.key) is ticket:
                    del self._pending_by_key[ticket.key]
                if rollback_request:
                    self._metrics.requests -= 1
                    self._metrics.misses -= 1
                self._metrics.admission_rejections += 1
                ticket.error = error
                ticket.state = "failed"
                ticket.storage_done.set()
                ticket.completed.set()
                self._condition.notify_all()
            if ticket.cuda_timeline is not None:
                ticket.cuda_timeline.cancel_transfer(
                    ticket,
                    "an async ticket was rejected before worker admission",
                )
            raise error from exc

    def _wait_for_storage(self, ticket: ExpertLoadTicket) -> None:
        started = time.perf_counter()
        while not ticket.storage_done.wait(timeout=0.002):
            self._poll_transfer_completions()
        elapsed = time.perf_counter() - started
        with self._condition:
            self._metrics.demand_wait_seconds += elapsed
        self._raise_ticket_error(ticket)

    def _raise_ticket_error(self, ticket: ExpertLoadTicket) -> None:
        error = ticket.error
        if error is None:
            return
        with self._condition:
            self._unobserved_errors.pop(id(ticket), None)
        raise error

    def _refresh_stale_resident_ticket(
        self,
        ticket: ExpertLoadTicket,
        *,
        cuda_timeline: _CudaTimelineCapture | None = None,
    ) -> ExpertLoadTicket:
        if ticket.state == "discarded":
            return self._requeue_discarded_lookahead(
                ticket,
                cuda_timeline=cuda_timeline,
            )
        if ticket.state != "resident":
            return ticket
        with self._condition:
            if self._closed or not self._accepting:
                raise RuntimeError("expert slot cache is closed")
            current_slot = self._key_to_slot.get(ticket.key)
            if current_slot is not None:
                ticket.destination_slot = current_slot
                ticket.destination_generation = self._slot_generation[current_slot]
                return ticket
            pending = self._pending_by_key.get(ticket.key)
            if ticket.counted_as_hit:
                self._metrics.hits -= 1
                self._metrics.misses += 1
                ticket.counted_as_hit = False
            if pending is not None:
                # This stale ticket already represents real demand (normally
                # a submit-time hit that was evicted before acquisition).  A
                # concurrent internal lookahead for the same key must inherit
                # that demand before it is returned; otherwise reservation
                # reclaim can mistake a lease-bound load for expendable work.
                if ticket.demanded:
                    pending.demanded = True
                    pending.request_clock = ticket.request_clock
                    self._admit_proactive_reservation_locked(pending)
                self._metrics.coalesced_requests += 1
                return pending
            ticket.queued_at = time.perf_counter()
            ticket.storage_done.clear()
            ticket.completed.clear()
            ticket.stage_index = None
            ticket.destination_slot = None
            ticket.destination_generation = 0
            ticket.bytes_read = 0
            ticket.error = None
            ticket.ready_event = None
            ticket.transfer_started_event = None
            ticket.transfer_enqueued.clear()
            ticket.reservation_active = False
            ticket.discard_requested = False
            # A resident placeholder has no transfer of its own.  Associate
            # it with the *current* capture only after eviction made this a
            # real requeue; the helper below reserves it before worker
            # visibility.  Rebinding also drops any completed old scope.
            ticket.cuda_timeline = cuda_timeline
            ticket.state = "queued"
            self._pending_by_key[ticket.key] = ticket
            self._metrics.pending_loads_peak = max(
                self._metrics.pending_loads_peak,
                len(self._pending_by_key),
            )
        self._enqueue_ticket_after_timeline_reservation(ticket)
        return ticket

    def _requeue_discarded_lookahead(
        self,
        ticket: ExpertLoadTicket,
        *,
        cuda_timeline: _CudaTimelineCapture | None = None,
    ) -> ExpertLoadTicket:
        """Turn a reclaimed internal lookahead into a fresh real-demand load."""

        with self._condition:
            if not ticket.demanded:  # pragma: no cover - caller invariant
                raise RuntimeError("discarded lookahead must be demanded before reuse")
            if self._closed or not self._accepting:
                raise RuntimeError("expert slot cache is closed")
            pending = self._pending_by_key.get(ticket.key)
            if pending is not None and pending is not ticket:
                self._metrics.coalesced_requests += 1
                return pending
            ticket.queued_at = time.perf_counter()
            ticket.storage_done.clear()
            ticket.completed.clear()
            ticket.stage_index = None
            ticket.destination_slot = None
            ticket.destination_generation = 0
            ticket.bytes_read = 0
            ticket.error = None
            ticket.ready_event = None
            ticket.transfer_started_event = None
            ticket.transfer_enqueued.clear()
            ticket.reservation_active = False
            ticket.discard_requested = False
            # A reclaimed lookahead may be demanded during a later model
            # call.  Bind the currently active scope (or deliberately none),
            # rather than retaining a capture that has already closed.
            ticket.cuda_timeline = cuda_timeline
            ticket.state = "queued"
            self._pending_by_key[ticket.key] = ticket
            self._metrics.pending_loads_peak = max(
                self._metrics.pending_loads_peak,
                len(self._pending_by_key),
            )
        self._enqueue_ticket_after_timeline_reservation(ticket)
        return ticket

    @property
    def resident_keys(self) -> tuple[ExpertKey, ...]:
        with self._lock:
            return tuple(sorted(self._key_to_slot))

    def _adaptive_should_use_async(
        self,
        layer: int,
        expert_ids: Iterable[int],
    ) -> bool:
        """Select async for a call that starts with free slots and misses.

        The first conservative selector deliberately avoids async eviction
        scheduling for calls that start with a full eligible layer partition.
        The RTX 6000 Ada study found that empty-cache passes usually benefited
        from async, while retained-cache and longer runs could regress.
        """
        keys = tuple(ExpertKey(layer, int(expert_id)) for expert_id in expert_ids)
        with self._condition:
            if self.staging_slots < 2:
                return False
            missing = tuple(key for key in keys if key not in self._key_to_slot)
            if len(missing) < 2:
                return False
            candidate_slots = {
                slot for key in missing for slot in self._async_candidate_slots(key)
            }
            return any(self._slot_to_key[slot] is None for slot in candidate_slots)

    def _record_adaptive_decision(self, use_async: bool, experts: int) -> None:
        with self._condition:
            if use_async:
                self._metrics.adaptive_async_forwards += 1
                self._metrics.adaptive_async_experts += experts
            else:
                self._metrics.adaptive_sync_forwards += 1
                self._metrics.adaptive_sync_experts += experts

    def _select_lru_slot(self, slots: tuple[int, ...]) -> int:
        reusable = [
            slot
            for slot in slots
            if not self._slot_copying[slot]
            and self._slot_pin_count[slot] == 0
            and self._slot_reservation[slot] is None
            and not self._slot_quarantined[slot]
        ]
        if not reusable:
            raise RuntimeError("no safely reusable expert cache slot is available")
        empty = [slot for slot in reusable if self._slot_to_key[slot] is None]
        if empty:
            return min(empty)
        return min(reusable, key=lambda slot: (self._last_used[slot], slot))

    def _select_slot(self, key: ExpertKey) -> int:
        static_slot = self._static_slots.get(key)
        if static_slot is not None:
            slot = static_slot
        else:
            dynamic_slots = self._dynamic_slots_by_layer[key.layer]
            if self.policy is CachePolicy.STATIC:
                slot = dynamic_slots[0]
            elif self.policy is CachePolicy.HYBRID:
                slot = self._select_lru_slot(dynamic_slots)
            else:
                slot = self._select_lru_slot(self._layer_slots[key.layer])
        if (
            self._slot_copying[slot]
            or self._slot_pin_count[slot] != 0
            or self._slot_reservation[slot] is not None
            or self._slot_quarantined[slot]
        ):
            raise RuntimeError("selected expert cache slot is not safely reusable")
        return slot

    def _async_candidate_slots(self, key: ExpertKey) -> tuple[int, ...]:
        static_slot = self._static_slots.get(key)
        if static_slot is not None:
            return (static_slot,)
        dynamic_slots = self._dynamic_slots_by_layer[key.layer]
        if self.policy is CachePolicy.STATIC:
            return (dynamic_slots[0],)
        if self.policy is CachePolicy.HYBRID:
            return dynamic_slots
        return self._layer_slots[key.layer]

    def _select_async_slot_locked(self, key: ExpertKey) -> int | None:
        available = tuple(
            slot
            for slot in self._async_candidate_slots(key)
            if not self._slot_copying[slot]
            and self._slot_pin_count[slot] == 0
            and self._slot_reservation[slot] is None
            and not self._slot_quarantined[slot]
        )
        if not available:
            return None
        empty = [slot for slot in available if self._slot_to_key[slot] is None]
        if empty:
            return min(empty)
        return min(available, key=lambda slot: (self._last_used[slot], slot))

    def _request_undemanded_reservation_reclaim_locked(self, key: ExpertKey) -> bool:
        """Ask one physical-only reservation to make room for real demand.

        A reservation is never an eviction target, but it must also not make a
        public or future demand wait forever.  A completed speculative copy is
        released immediately; an in-flight one is marked for release after its
        CUDA event completes.  No logical cache entry is changed here.
        """

        for slot in self._async_candidate_slots(key):
            reserved = self._slot_reservation[slot]
            if reserved is None or not reserved.is_lookahead or reserved.demanded:
                continue
            # Demand promotion removes the reservation before creating a
            # lease.  Keep this defensive gate as well: even a malformed or
            # future state transition must never reclaim a buffer in use.
            if self._slot_pin_count[slot] != 0:
                continue
            reserved.discard_requested = True
            if reserved.state == "prefetched":
                return self._retire_undemanded_lookahead_locked(reserved)
            if reserved.state == "prefetching":
                return True
        return False

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _copy_staging_to_slot(
        self,
        slot: int,
        staging_gate_up: torch.Tensor,
        staging_down: torch.Tensor,
        *,
        cuda_timeline: _CudaTimelineCapture | None = None,
        key: ExpertKey | None = None,
    ) -> None:
        started_event = None
        ready_event = None
        if cuda_timeline is not None:
            if self.device.type != "cuda" or key is None:
                raise RuntimeError("CUDA timeline capture requires a CUDA expert key")
            stream = torch.cuda.current_stream(self.device)
            cuda_timeline.wait_on_origin(stream)
            started_event = torch.cuda.Event(enable_timing=True)
            ready_event = torch.cuda.Event(enable_timing=True)
            started_event.record(stream)
        self._gate_up_slots[slot].copy_(staging_gate_up, non_blocking=False)
        self._down_slots[slot].copy_(staging_down, non_blocking=False)
        if ready_event is not None:
            ready_event.record(torch.cuda.current_stream(self.device))
        self._synchronize_device()
        if started_event is not None and ready_event is not None and key is not None:
            cuda_timeline.record_transfer(key, started_event, ready_event)

    def _enqueue_staging_to_slot(
        self,
        slot: int,
        staging_gate_up: torch.Tensor,
        staging_down: torch.Tensor,
        wait_events: tuple[Any, ...],
    ) -> tuple[Any, Any]:
        stream = self._transfer_stream
        if stream is None:  # pragma: no cover - caller/device invariant
            raise RuntimeError("CUDA transfer stream is unavailable")
        started = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event(enable_timing=True)
        with self._transfer_submit_lock, torch.cuda.stream(stream):
            for event in wait_events:
                stream.wait_event(event)
            started.record(stream)
            self._gate_up_slots[slot].copy_(
                staging_gate_up,
                non_blocking=True,
            )
            self._down_slots[slot].copy_(
                staging_down,
                non_blocking=True,
            )
            ready.record(stream)
        return started, ready

    def _complete_transfer_locked(self, ticket: ExpertLoadTicket) -> None:
        slot = ticket.destination_slot
        stage_index = ticket.stage_index
        if slot is None or stage_index is None:
            raise RuntimeError("incomplete async transfer ticket")
        if self._slot_generation[slot] != ticket.destination_generation:
            raise RuntimeError("stale async transfer completion")
        self._slot_copying[slot] = False
        if self.device.type == "cuda":
            self._metrics.host_to_device_bytes += self.store.spec.size_bytes
        self._metrics.transfer_loads += 1

        if ticket.reservation_active:
            if ticket.demanded:  # pragma: no cover - admission lock invariant
                raise RuntimeError("demanded lookahead was not logically admitted")
            if self._slot_reservation[slot] is not ticket:
                raise RuntimeError("prefetch completion lost its reservation")
            # Keep the physical slot, but do not create a logical cache entry.
            # Its staging buffer can finally be reused because the CUDA event
            # has completed; a later demand will atomically promote this slot.
            ticket.state = "prefetched"
            self._free_staging_slot_locked(stage_index, ticket)
            if ticket.discard_requested:
                self._retire_undemanded_lookahead_locked(ticket)
            self._condition.notify_all()
            return

        existing_key = self._slot_to_key[slot]
        if existing_key is None:
            self._slot_to_key[slot] = ticket.key
            self._key_to_slot[ticket.key] = slot
            self._last_used[slot] = ticket.request_clock
        elif existing_key != ticket.key:
            raise RuntimeError("async transfer destination has a different expert")
        if self._pending_by_key.get(ticket.key) is ticket:
            del self._pending_by_key[ticket.key]
        ticket.state = "resident"
        ticket.completed.set()
        self._free_staging_slot_locked(stage_index, ticket)
        self._condition.notify_all()

    def _fail_transfer(self, ticket: ExpertLoadTicket, error: Exception) -> None:
        if ticket.cuda_timeline is not None:
            ticket.cuda_timeline.cancel_transfer(
                ticket,
                f"async H2D failed: {error}",
            )
            # An event pair may already have been committed before a later
            # readiness/query failure reaches this boundary.  Preserve the
            # fail-closed contract in that case too.
            ticket.cuda_timeline.invalidate(f"async H2D failed: {error}")
        if self.device.type == "cuda" and self._transfer_stream is not None:
            try:
                self._transfer_stream.synchronize()
            except Exception as synchronize_error:  # noqa: BLE001
                with self._condition:
                    if self._pipeline_error is None:
                        self._pipeline_error = synchronize_error
        with self._condition:
            slot = ticket.destination_slot
            if (
                slot is not None
                and self._slot_generation[slot] == ticket.destination_generation
            ):
                self._slot_copying[slot] = False
                if self._slot_reservation[slot] is ticket:
                    self._slot_reservation[slot] = None
                    ticket.reservation_active = False
                if self._slot_to_key[slot] == ticket.key:
                    del self._key_to_slot[ticket.key]
                self._slot_to_key[slot] = None
                self._last_used[slot] = 0
                if self._slot_pin_count[slot] > 0:
                    # Do not reuse a physical buffer while a lease may still
                    # be executing on a compute stream.  The release path
                    # records that stream's tail event and only then returns
                    # the slot to the allocators.
                    self._slot_quarantined[slot] = True
                else:
                    self._slot_quarantined[slot] = False
                    # A lease can record its compute-stream tail before a
                    # later poll reports this H2D as failed.  Keep that event
                    # even though the logical mapping is gone: a subsequent
                    # transfer must still wait before overwriting the buffer.
            if self._pending_by_key.get(ticket.key) is ticket:
                del self._pending_by_key[ticket.key]
            if ticket.stage_index is not None:
                self._free_staging_slot_locked(ticket.stage_index, ticket)
            ticket.error = error
            ticket.state = "failed"
            ticket.completed.set()
            ticket.transfer_enqueued.set()
            self._unobserved_errors[id(ticket)] = error
            self._metrics.transfer_failures += 1
            self._condition.notify_all()

    def _poll_transfer_completions(self) -> None:
        if not self._uses_async_infrastructure or self.device.type != "cuda":
            return
        with self._condition:
            inflight = tuple(
                ticket
                for ticket, state in zip(
                    self._stage_owner,
                    self._stage_state,
                    strict=True,
                )
                if ticket is not None and state == "h2d"
            )
        if not inflight:
            return
        try:
            # Event operations consult thread-local CUDA device state when
            # called from the I/O worker as it waits for staging capacity.
            torch.cuda.set_device(self.device)
        except Exception as exc:  # noqa: BLE001 - CUDA worker boundary
            with self._condition:
                self._pipeline_error = exc
            # A failed device selection must terminally wake every waiter.  In
            # particular, wait_idle()/close() cannot leave a staged ticket in
            # an eternal h2d state merely because event polling is unavailable.
            for ticket in inflight:
                self._fail_transfer(ticket, exc)
            return
        for ticket in inflight:
            ready = ticket.ready_event
            if ready is None:
                continue
            try:
                completed = bool(ready.query())
            except Exception as exc:  # noqa: BLE001 - CUDA failure boundary
                with self._condition:
                    self._pipeline_error = exc
                self._fail_transfer(ticket, exc)
                continue
            if not completed:
                continue
            transfer_seconds = 0.0
            started = ticket.transfer_started_event
            if started is not None:
                try:
                    transfer_seconds = float(started.elapsed_time(ready)) / 1000.0
                except RuntimeError:
                    transfer_seconds = 0.0
            with self._condition:
                if ticket.state not in ("copying", "prefetching"):
                    continue
                self._metrics.transfer_seconds += transfer_seconds
                self._complete_transfer_locked(ticket)

    def _schedule_ticket_transfer(
        self,
        ticket: ExpertLoadTicket,
        _use_stream: Any | None,
    ) -> None:
        wait_events: list[Any] = []
        cuda_timeline = ticket.cuda_timeline
        while True:
            self._poll_transfer_completions()
            with self._condition:
                if ticket.destination_slot is not None:
                    return
                if self._pipeline_error is not None:
                    raise RuntimeError("asynchronous expert pipeline failed") from (
                        self._pipeline_error
                    )
                slot = self._select_async_slot_locked(ticket.key)
                if slot is None:
                    self._request_undemanded_reservation_reclaim_locked(ticket.key)
                    # An in-flight speculative H2D retains staging until its
                    # completion event; wait outside all CUDA work, then poll
                    # again to retire the physical-only reservation safely.
                    self._condition.wait(timeout=0.002)
                    continue

                # Selection and claim happen under one condition critical
                # section.  The I/O worker cannot reserve this empty slot in
                # the former select-to-enqueue gap.
                evicted_key = self._slot_to_key[slot]
                # A failed transfer can have already removed the logical key
                # while a previously leased compute stream is still using the
                # physical slot.  Preserve that stream ordering even when the
                # slot is logically empty.
                if self.device.type == "cuda":
                    last_use = self._slot_last_use_event[slot]
                    if last_use is not None:
                        wait_events.append(last_use)

                if evicted_key is not None:
                    del self._key_to_slot[evicted_key]
                    self._slot_to_key[slot] = None
                    self._last_used[slot] = 0
                    self._metrics.evictions += 1
                self._slot_generation[slot] += 1
                ticket.destination_slot = slot
                ticket.destination_generation = self._slot_generation[slot]
                ticket.state = "copying"
                ticket.transfer_enqueued.clear()
                self._slot_copying[slot] = True
                stage_index = ticket.stage_index
                if stage_index is None:  # pragma: no cover - storage invariant
                    raise RuntimeError("async ticket has no staging slot")
                self._stage_state[stage_index] = "h2d"
                staging_gate_up = self._staging_gate_up[stage_index]
                staging_down = self._staging_down[stage_index]
                break

        transfer_started = time.perf_counter()
        timeline_claimed = False
        try:
            if self.device.type == "cuda":
                if cuda_timeline is not None:
                    timeline_claimed = cuda_timeline.claim_transfer(ticket)
                    if timeline_claimed:
                        if (
                            self._transfer_stream is None
                        ):  # pragma: no cover - invariant
                            raise RuntimeError("CUDA transfer stream is unavailable")
                        # Submit the origin dependency under the same transfer
                        # lock as timing/copy events, including demand fallback.
                        wait_events.append(cuda_timeline.origin_event)
                started_event, ready_event = self._enqueue_staging_to_slot(
                    slot,
                    staging_gate_up,
                    staging_down,
                    tuple(wait_events),
                )
                with self._condition:
                    ticket.transfer_started_event = started_event
                    ticket.ready_event = ready_event
                    ticket.transfer_enqueued.set()
                    self._condition.notify_all()
                if timeline_claimed:
                    cuda_timeline.commit_transfer(ticket, started_event, ready_event)
            else:
                self._copy_staging_to_slot(slot, staging_gate_up, staging_down)
                with self._condition:
                    self._metrics.transfer_seconds += (
                        time.perf_counter() - transfer_started
                    )
                    self._complete_transfer_locked(ticket)
                    ticket.transfer_enqueued.set()
        except Exception as exc:  # noqa: BLE001 - transactional rollback
            if timeline_claimed:
                cuda_timeline.cancel_transfer(
                    ticket,
                    f"demand H2D enqueue failed: {exc}",
                )
            self._fail_transfer(ticket, exc)
            self._raise_ticket_error(ticket)

    def _admit_proactive_reservation_locked(
        self,
        ticket: ExpertLoadTicket,
    ) -> None:
        """Publish an already-reserved lookahead only at actual demand time."""

        if not ticket.reservation_active:
            return
        slot = ticket.destination_slot
        if slot is None:  # pragma: no cover - reservation invariant
            raise RuntimeError("reserved lookahead has no destination slot")
        if self._slot_reservation[slot] is not ticket:
            raise RuntimeError("reserved lookahead lost its physical slot")
        if self._slot_to_key[slot] is not None:
            raise RuntimeError("reserved lookahead slot is already logical")
        existing_slot = self._key_to_slot.get(ticket.key)
        if existing_slot is not None:
            raise RuntimeError("reserved lookahead key is already logical")

        # The slot was verified empty when reserved; promotion therefore makes
        # no eviction and changes no request accounting.  The demand code
        # surrounding this call owns the miss and LRU timestamp.
        self._slot_reservation[slot] = None
        ticket.reservation_active = False
        self._slot_to_key[slot] = ticket.key
        self._key_to_slot[ticket.key] = slot
        self._last_used[slot] = ticket.request_clock
        if ticket.state == "prefetched":
            ticket.state = "resident"
            if self._pending_by_key.get(ticket.key) is ticket:
                del self._pending_by_key[ticket.key]
            ticket.completed.set()
        elif ticket.state == "prefetching":
            ticket.state = "copying"
        elif ticket.state != "copying":  # pragma: no cover - state invariant
            raise RuntimeError("reserved lookahead is not ready to admit")
        self._condition.notify_all()

    def _wait_for_transfer_enqueue(self, ticket: ExpertLoadTicket) -> None:
        """Wait only for a worker to publish its CUDA completion event."""

        started = time.perf_counter()
        while not ticket.transfer_enqueued.wait(timeout=0.002):
            self._poll_transfer_completions()
        elapsed = time.perf_counter() - started
        with self._condition:
            self._metrics.demand_wait_seconds += elapsed
        self._raise_ticket_error(ticket)

    def _account_demand(self, ticket: ExpertLoadTicket) -> None:
        with self._condition:
            self._clock += 1
            self._metrics.requests += 1
            ticket.request_clock = self._clock
            ticket.demanded = True
            existing_slot = self._key_to_slot.get(ticket.key)
            if existing_slot is None:
                self._metrics.misses += 1
                self._admit_proactive_reservation_locked(ticket)
                return
            self._metrics.hits += 1
            self._last_used[existing_slot] = self._clock

    def _acquire_ticket(
        self,
        ticket: ExpertLoadTicket,
        *,
        compute_stream: Any | None,
        synchronize: bool,
        account_demand: bool = False,
        cuda_timeline: _CudaTimelineCapture | None = None,
    ) -> _ExpertLease:
        if account_demand:
            self._account_demand(ticket)
        ticket = self._refresh_stale_resident_ticket(
            ticket,
            cuda_timeline=cuda_timeline,
        )
        self._raise_ticket_error(ticket)
        if ticket.state not in ("resident", "copying"):
            self._wait_for_storage(ticket)
            self._schedule_ticket_transfer(ticket, compute_stream)
        elif ticket.state == "copying" and ticket.ready_event is None:
            self._wait_for_transfer_enqueue(ticket)
        self._raise_ticket_error(ticket)

        ready = ticket.ready_event
        if ready is not None:
            if synchronize:
                try:
                    ready.synchronize()
                except Exception as exc:  # noqa: BLE001 - CUDA failure boundary
                    with self._condition:
                        self._pipeline_error = exc
                    self._fail_transfer(ticket, exc)
                    self._raise_ticket_error(ticket)
                self._poll_transfer_completions()
            else:
                if compute_stream is None:  # pragma: no cover - caller invariant
                    raise RuntimeError("compute stream is required for async acquire")
                compute_stream.wait_event(ready)

        slot = ticket.destination_slot
        if slot is None:  # pragma: no cover - ticket invariant
            raise RuntimeError("async ticket has no destination slot")
        with self._condition:
            self._raise_ticket_error(ticket)
            if self._slot_generation[slot] != ticket.destination_generation:
                raise RuntimeError("async expert lease became stale")
            if ticket.state == "resident" and self._slot_to_key[slot] != ticket.key:
                raise RuntimeError("reserved async expert is no longer resident")
            self._slot_pin_count[slot] += 1
            weights = ExpertWeights(
                self._gate_up_slots[slot],
                self._down_slots[slot],
            )
        return _ExpertLease(
            cache=self,
            slot=slot,
            generation=ticket.destination_generation,
            weights=weights,
        )

    def _release_lease(
        self,
        slot: int,
        generation: int,
        stream: Any | None,
    ) -> None:
        use_event = None
        if self.device.type == "cuda":
            if stream is None:
                stream = torch.cuda.current_stream(self.device)
            use_event = torch.cuda.Event(enable_timing=False)
            use_event.record(stream)
        with self._condition:
            if self._slot_generation[slot] != generation:
                raise RuntimeError("cannot release a stale expert lease")
            if self._slot_pin_count[slot] <= 0:
                raise RuntimeError("expert lease was not pinned")
            self._slot_pin_count[slot] -= 1
            if use_event is not None:
                self._slot_last_use_event[slot] = use_event
            if self._slot_pin_count[slot] == 0 and self._slot_quarantined[slot]:
                # `_fail_transfer` already removed the logical mapping.  Keep
                # the just-recorded last-use event so the next H2D waits for
                # the outstanding compute before overwriting this buffer.
                self._slot_quarantined[slot] = False
                self._slot_copying[slot] = False
                self._slot_to_key[slot] = None
                self._last_used[slot] = 0
            self._condition.notify_all()

    def resolve(self, ticket: ExpertLoadTicket) -> ExpertWeights:
        """Wait for a submitted expert and return fully materialized weights."""
        if self.pipeline_mode != "async":
            raise RuntimeError("resolve requires pipeline_mode='async'")
        if self.device.type == "cuda":
            raise RuntimeError(
                "raw CUDA weights are unsafe in async mode; use PagedExpertRuntime"
            )
        with self.execution_lock:
            stream = (
                torch.cuda.current_stream(self.device)
                if self.device.type == "cuda"
                else None
            )
            lease = self._acquire_ticket(
                ticket,
                compute_stream=stream,
                synchronize=True,
            )
            weights = lease.weights
            lease.release_after(stream)
            return weights

    def get(
        self,
        key: ExpertKey,
        *,
        cuda_timeline: _CudaTimelineCapture | None = None,
    ) -> ExpertWeights:
        """Return weights valid until a later miss reuses their cache slot."""
        if key not in self.store:
            raise KeyError(f"unknown expert: {key.compact()}")
        if self.pipeline_mode == "async":
            if self.device.type == "cuda":
                raise RuntimeError(
                    "raw CUDA weights are unsafe in async mode; use PagedExpertRuntime"
                )
            with self.execution_lock:
                return self.resolve(self.submit(key))
        with self.execution_lock, self._lock:
            if self._closed:
                raise RuntimeError("expert slot cache is closed")
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
            last_use = self._slot_last_use_event[slot]

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
            self._metrics.storage_loads += 1

            if evicted_key is not None or last_use is not None:
                # In sync mode there is no transfer stream on which to wait
                # for a prior lease's tail event.  Synchronize before copying
                # even if an earlier failure made the slot logically empty.
                self._synchronize_device()
            if evicted_key is not None:
                del self._key_to_slot[evicted_key]
                self._slot_to_key[slot] = None
                self._last_used[slot] = 0
                self._metrics.evictions += 1

            transfer_started = time.perf_counter()
            try:
                if cuda_timeline is None:
                    self._copy_staging_to_slot(slot, staging_gate_up, staging_down)
                else:
                    self._copy_staging_to_slot(
                        slot,
                        staging_gate_up,
                        staging_down,
                        cuda_timeline=cuda_timeline,
                        key=key,
                    )
            except BaseException:
                self._slot_to_key[slot] = None
                self._last_used[slot] = 0
                raise
            finally:
                self._metrics.transfer_seconds += time.perf_counter() - transfer_started
            if self.device.type != "cpu":
                self._metrics.host_to_device_bytes += self.store.spec.size_bytes
            self._metrics.transfer_loads += 1

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
                if self.pipeline_mode == "async":
                    stream = (
                        torch.cuda.current_stream(self.device)
                        if self.device.type == "cuda"
                        else None
                    )
                    lease = self._acquire_ticket(
                        self.submit(key),
                        compute_stream=stream,
                        synchronize=True,
                    )
                    lease.release_after(stream)
                else:
                    self.get(key)

    def wait_idle(self) -> None:
        """Drain submitted async work without closing the external store."""
        if not self._uses_async_infrastructure:
            return
        first_error: Exception | None = None
        with self.execution_lock:
            while True:
                self._poll_transfer_completions()
                with self._condition:
                    unobserved_errors = tuple(self._unobserved_errors.values())
                    self._unobserved_errors.clear()
                    pending = tuple(self._pending_by_key.values())
                    stages_idle = all(state == "free" for state in self._stage_state)
                    idle = not pending and stages_idle and self._queued_job_count == 0
                if first_error is None and unobserved_errors:
                    first_error = unobserved_errors[0]
                if idle:
                    break
                for ticket in pending:
                    with self._condition:
                        speculative = ticket.is_lookahead and not ticket.demanded
                        retired = (
                            self._retire_undemanded_lookahead_locked(ticket)
                            if speculative
                            else False
                        )
                        state = ticket.state
                    if speculative:
                        if retired:
                            continue
                        try:
                            if state in ("queued", "reading"):
                                ticket.storage_done.wait(timeout=0.002)
                            elif state == "prefetching":
                                self._wait_for_transfer_enqueue(ticket)
                                ready = ticket.ready_event
                                if ready is not None:
                                    ready.synchronize()
                                self._poll_transfer_completions()
                            elif state == "prefetched":
                                # A defensive pin gate can defer retirement
                                # until a concurrent lease has recorded its
                                # stream-tail event and released the slot.
                                with self._condition:
                                    self._condition.wait(timeout=0.002)
                            else:  # pragma: no cover - state invariant
                                raise RuntimeError(
                                    "undemanded lookahead did not make progress"
                                )
                        except Exception as exc:  # noqa: BLE001 - drain all tickets
                            if first_error is None:
                                first_error = exc
                        continue
                    try:
                        stream = (
                            torch.cuda.current_stream(self.device)
                            if self.device.type == "cuda"
                            else None
                        )
                        lease = self._acquire_ticket(
                            ticket,
                            compute_stream=stream,
                            synchronize=True,
                        )
                        lease.release_after(stream)
                    except Exception as exc:  # noqa: BLE001 - drain all tickets
                        if first_error is None:
                            first_error = exc
                        with self._condition:
                            still_pending = (
                                self._pending_by_key.get(ticket.key) is ticket
                            )
                        if still_pending:
                            # A global pipeline failure (for example CUDA
                            # device selection) makes this ready ticket
                            # impossible to materialize.  Terminally release
                            # it so wait_idle()/close() can finish draining.
                            self._fail_transfer(ticket, exc)
                self._poll_transfer_completions()
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Idempotently stop the optional worker; the store remains external."""
        with self._close_lock:
            if self._closed:
                return
            first_error: Exception | None = None
            with self.execution_lock:
                with self._condition:
                    self._accepting = False
                    self._condition.notify_all()
                try:
                    self.wait_idle()
                except Exception as exc:  # noqa: BLE001 - shutdown must continue
                    first_error = exc
                worker = self._worker
                jobs = self._jobs
            if worker is not None and jobs is not None:
                jobs.join()
                jobs.put(None)
                worker.join()
            if self._transfer_stream is not None:
                try:
                    self._transfer_stream.synchronize()
                except Exception as exc:  # noqa: BLE001 - shutdown must continue
                    if first_error is None:
                        first_error = exc
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            if first_error is not None:
                raise first_error

    def metrics(self) -> PagedRuntimeMetrics:
        with self._lock:
            return self._metrics.snapshot()

    def add_forward_seconds(self, elapsed: float) -> None:
        with self._lock:
            self._metrics.forward_seconds += elapsed


class PagedExpertRuntime:
    """Execute routed gated-MLP experts through a bounded slot cache."""

    def __init__(self, cache: ExpertSlotCache) -> None:
        self.cache = cache
        self._forward_lock = cache.execution_lock
        self._active_cuda_timeline: _CudaTimelineCapture | None = None

    def cuda_timeline_capture(self) -> _CudaTimelineCaptureScope:
        """Capture one model-call H2D/expert-compute CUDA timeline.

        The scope must include the caller's model invocation and its normal CUDA
        synchronization.  It is deliberately opt-in because CUDA timing events
        are instrumentation, not part of the default runtime data path.
        """

        return _CudaTimelineCaptureScope(self)

    def _begin_cuda_timeline_capture(self) -> _CudaTimelineCapture:
        with self._forward_lock:
            if self.cache.device.type != "cuda":
                raise RuntimeError("CUDA timeline capture requires a CUDA runtime")
            if self._active_cuda_timeline is not None:
                raise RuntimeError("a CUDA timeline capture is already active")
            # A capture owns one origin and one finite worker ledger.  Drain
            # work submitted before the scope so a prior lookahead cannot
            # enqueue an unrelated H2D into this call's evidence window.
            self.cache.wait_idle()
            transfer_loads_baseline = self.cache.metrics().transfer_loads
            capture = _CudaTimelineCapture(
                self.cache.device,
                transfer_loads_baseline=transfer_loads_baseline,
            )
            capture.begin(torch.cuda.current_stream(self.cache.device))
            self._active_cuda_timeline = capture
            return capture

    def _timeline_transfer_loads_after_synchronize(self) -> int:
        """Credit ready async H2Ds before finalizing one capture's coverage."""

        # `finish()` has synchronized the capture device but intentionally
        # leaves normal cache ownership alone.  Polling is enough to turn
        # those completed events into `transfer_loads`; a full `wait_idle()`
        # would retire valid undemanded proactive lookaheads and alter the
        # capture's logical lifecycle.
        self.cache._poll_transfer_completions()
        return self.cache.metrics().transfer_loads

    def _end_cuda_timeline_capture(
        self,
        capture: _CudaTimelineCapture,
        *,
        failed: bool,
    ) -> None:
        with self._forward_lock:
            if self._active_cuda_timeline is not capture:
                raise RuntimeError(
                    "CUDA timeline capture does not belong to this runtime"
                )
            try:
                if failed:
                    capture.abort()
                else:
                    capture.finish(
                        transfer_loads_after_synchronize=(
                            self._timeline_transfer_loads_after_synchronize
                        )
                    )
            finally:
                self._active_cuda_timeline = None

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
        expert_ids: list[int] = []
        for expert in active_experts:
            expert_id = int(expert)
            if expert_id == sentinel:
                continue
            if expert_id not in valid_experts:
                raise KeyError(f"unknown expert: L{layer}:E{expert_id}")
            expert_ids.append(expert_id)

        compute_stream = (
            torch.cuda.current_stream(hidden_states.device)
            if hidden_states.device.type == "cuda"
            else None
        )
        cuda_timeline = self._active_cuda_timeline
        use_async = self.cache.pipeline_mode == "async"
        if self.cache.pipeline_mode == "adaptive" and expert_ids:
            # A previous external submission must not share staging with the
            # synchronous branch selected for this forward.
            self.cache.wait_idle()
            use_async = self.cache._adaptive_should_use_async(layer, expert_ids)
            self.cache._record_adaptive_decision(use_async, len(expert_ids))
        tickets: dict[int, ExpertLoadTicket] = {}
        next_to_submit = 0
        try:
            if use_async:
                initial_depth = min(self.cache.staging_slots, len(expert_ids))
                while next_to_submit < initial_depth:
                    expert_id = expert_ids[next_to_submit]
                    next_key = ExpertKey(layer, expert_id)
                    tickets[expert_id] = (
                        self.cache._submit_lookahead(next_key)
                        if cuda_timeline is None
                        else self.cache._submit_lookahead(
                            next_key,
                            cuda_timeline=cuda_timeline,
                        )
                    )
                    next_to_submit += 1
            for expert_id in expert_ids:
                lease: _ExpertLease | None = None
                if use_async:
                    lease = self.cache._acquire_ticket(
                        tickets[expert_id],
                        compute_stream=compute_stream,
                        synchronize=False,
                        account_demand=True,
                        cuda_timeline=cuda_timeline,
                    )
                    weights = lease.weights
                else:
                    demand_key = ExpertKey(layer, expert_id)
                    weights = (
                        self.cache.get(demand_key)
                        if cuda_timeline is None
                        else self.cache.get(
                            demand_key,
                            cuda_timeline=cuda_timeline,
                        )
                    )
                compute_started = None
                try:
                    if use_async and next_to_submit < len(expert_ids):
                        next_expert = expert_ids[next_to_submit]
                        next_key = ExpertKey(layer, next_expert)
                        tickets[next_expert] = (
                            self.cache._submit_lookahead(next_key)
                            if cuda_timeline is None
                            else self.cache._submit_lookahead(
                                next_key,
                                cuda_timeline=cuda_timeline,
                            )
                        )
                        next_to_submit += 1
                    if cuda_timeline is not None:
                        if compute_stream is None:  # pragma: no cover - invariant
                            raise RuntimeError(
                                "CUDA timeline capture has no compute stream"
                            )
                        compute_started = cuda_timeline.begin_compute(
                            ExpertKey(layer, expert_id),
                            compute_stream,
                        )
                    token_index, top_k_position = torch.where(top_k_index == expert_id)
                    current_state = hidden_states[token_index]
                    gate, up = torch_functional.linear(
                        current_state, weights.gate_up
                    ).chunk(2, dim=-1)
                    current = torch_functional.silu(gate) * up
                    current = torch_functional.linear(current, weights.down)
                    current = current * top_k_weights[token_index, top_k_position, None]
                    final_hidden_states.index_add_(
                        0, token_index, current.to(final_hidden_states.dtype)
                    )
                finally:
                    if compute_started is not None:
                        cuda_timeline.end_compute(
                            ExpertKey(layer, expert_id),
                            compute_started,
                            compute_stream,
                        )
                    if lease is not None:
                        lease.release_after(compute_stream)
        except Exception:
            if use_async:
                try:
                    self.cache.wait_idle()
                except Exception as cleanup_error:  # noqa: BLE001
                    with self.cache._condition:
                        if self.cache._pipeline_error is None:
                            self.cache._pipeline_error = cleanup_error
            raise
        if hidden_states.device.type == "cuda":
            if use_async:
                try:
                    compute_stream.synchronize()
                except Exception as exc:
                    with self.cache._condition:
                        self.cache._pipeline_error = exc
                    raise
                self.cache._poll_transfer_completions()
            else:
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


@dataclass(frozen=True, slots=True)
class TransformersMoEAdapter:
    """Describe one compatible Transformers sparse-MoE module layout.

    OLMoE and Mixtral expose the same packed expert backend in the pinned
    Transformers release, but their model identities remain explicit.  Keeping
    that identity in a small adapter prevents model-specific checks from
    leaking into the cache, storage and scheduling layers.
    """

    model_type: str
    display_name: str
    expert_intermediate_size_attr: str = "intermediate_size"

    def layers(self, model: Any) -> Any:
        try:
            return model.model.layers
        except AttributeError as exc:
            raise TypeError(
                f"expected a {self.display_name}ForCausalLM-compatible model"
            ) from exc

    def experts(self, layer_module: Any) -> Any:
        try:
            return layer_module.mlp.experts
        except AttributeError as exc:
            raise TypeError(
                f"expected {self.display_name} decoder layers with mlp.experts"
            ) from exc

    def expert_intermediate_size(self, config: Any) -> int:
        value = getattr(config, self.expert_intermediate_size_attr, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(
                f"{self.display_name} config has no positive "
                f"{self.expert_intermediate_size_attr}"
            )
        return value


_TRANSFORMERS_MOE_ADAPTERS = {
    "olmoe": TransformersMoEAdapter("olmoe", "OLMoE"),
    "mixtral": TransformersMoEAdapter("mixtral", "Mixtral"),
    "qwen2_moe": TransformersMoEAdapter(
        "qwen2_moe",
        "Qwen2MoE",
        expert_intermediate_size_attr="moe_intermediate_size",
    ),
}


def transformers_moe_adapter_for_model(model: Any) -> TransformersMoEAdapter:
    """Resolve a fail-closed adapter for one supported Transformers model."""

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str):
        raise TypeError("Transformers MoE model config has no model_type")
    try:
        return _TRANSFORMERS_MOE_ADAPTERS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_TRANSFORMERS_MOE_ADAPTERS))
        raise TypeError(
            f"unsupported Transformers MoE model type {model_type!r}; "
            f"supported: {supported}"
        ) from exc


def attach_transformers_moe_runtime(
    model: Any,
    runtime: PagedExpertRuntime,
    *,
    implementation: str = "moevm_paged",
) -> TransformersMoEAdapter:
    """Attach the paged backend to one supported Transformers sparse MoE."""

    adapter = transformers_moe_adapter_for_model(model)
    register_transformers_paged_experts(implementation)
    layers = adapter.layers(model)
    config = model.config
    store = runtime.cache.store
    expected_layers = tuple(range(len(layers)))
    if store.layers != expected_layers:
        raise ValueError("model layer count does not match the expert store")
    if config.hidden_act != "silu":
        raise ValueError(
            f"paged {adapter.display_name} requires SiLU, got {config.hidden_act}"
        )
    if (
        config.hidden_size != store.spec.hidden_size
        or adapter.expert_intermediate_size(config) != store.spec.intermediate_size
    ):
        raise ValueError("model dimensions do not match the expert store")
    for layer_index, layer_module in enumerate(layers):
        experts = adapter.experts(layer_module)
        if experts.num_experts != len(store.experts_in_layer(layer_index)):
            raise ValueError(
                f"model expert count does not match store layer {layer_index}"
            )
        if experts.gate_up_proj.dtype != store.spec.dtype:
            raise ValueError(
                f"model expert dtype does not match store layer {layer_index}"
            )
        expected_gate_up_shape = (
            experts.num_experts,
            2 * store.spec.intermediate_size,
            store.spec.hidden_size,
        )
        expected_down_shape = (
            experts.num_experts,
            store.spec.hidden_size,
            store.spec.intermediate_size,
        )
        if tuple(experts.gate_up_proj.shape) != expected_gate_up_shape:
            raise ValueError(
                f"model gate/up expert shape does not match store layer {layer_index}"
            )
        if tuple(experts.down_proj.shape) != expected_down_shape:
            raise ValueError(
                f"model down expert shape does not match store layer {layer_index}"
            )
    config._experts_implementation = implementation
    for layer_index, layer_module in enumerate(layers):
        experts = adapter.experts(layer_module)
        experts._moevm_paged_runtime = runtime
        experts._moevm_layer_index = layer_index
        experts.config._experts_implementation = implementation
    return adapter


def attach_transformers_olmoe_runtime(
    model: Any,
    runtime: PagedExpertRuntime,
    *,
    implementation: str = "moevm_paged",
) -> None:
    """Backward-compatible OLMoE-specific attachment entry point."""

    adapter = transformers_moe_adapter_for_model(model)
    if adapter.model_type != "olmoe":
        raise TypeError("expected an OLMoEForCausalLM-compatible model")
    attach_transformers_moe_runtime(
        model,
        runtime,
        implementation=implementation,
    )


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
    """Fail closed if a supported paged model has unsafe meta or dtype state."""
    adapter = transformers_moe_adapter_for_model(model)
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
    for layer_index, layer in enumerate(adapter.layers(model)):
        experts = adapter.experts(layer)
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
        "adapter": adapter.model_type,
        "expert_meta_parameters": expert_meta,
        "non_expert_parameters": non_expert_parameters,
        "buffers": sum(1 for _ in model.named_buffers()),
        "cpu_buffers_allowed": sum(
            1 for _, buffer in model.named_buffers() if buffer.device.type == "cpu"
        ),
        "dtype": str(store.spec.dtype),
    }
