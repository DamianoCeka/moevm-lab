from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .trace import read_trace
from .types import RoutingStep

COLD_LRU = "cold_lru"
STATIC_HOT = "static_hot"
HYBRID_HOT_LRU = "hybrid_hot_lru"
PLACEMENT_POLICIES = (COLD_LRU, STATIC_HOT, HYBRID_HOT_LRU)


@dataclass(frozen=True, slots=True)
class PlacementTrace:
    """A routing trace with the identity required for leakage checks."""

    workload_id: str
    source: str
    sha256: str
    steps: tuple[RoutingStep, ...]


@dataclass(slots=True)
class _Counts:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def add(self, *, hit: bool, evicted: bool) -> None:
        self.accesses += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        if evicted:
            self.evictions += 1


class _ColdLru:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: OrderedDict[int, None] = OrderedDict()

    def access(self, expert: int) -> tuple[bool, bool]:
        if expert in self.entries:
            self.entries.move_to_end(expert)
            return True, False
        if self.capacity == 0:
            return False, False
        evicted = len(self.entries) == self.capacity
        if evicted:
            self.entries.popitem(last=False)
        self.entries[expert] = None
        return False, evicted


class _StaticHot:
    def __init__(self, protected: frozenset[int]) -> None:
        self.protected = protected

    def access(self, expert: int) -> tuple[bool, bool]:
        return expert in self.protected, False


class _HybridHotLru:
    def __init__(self, protected: frozenset[int], dynamic_capacity: int) -> None:
        self.protected = protected
        self.dynamic_capacity = dynamic_capacity
        self.dynamic: OrderedDict[int, None] = OrderedDict()

    def access(self, expert: int) -> tuple[bool, bool]:
        if expert in self.protected:
            return True, False
        if expert in self.dynamic:
            self.dynamic.move_to_end(expert)
            return True, False
        if self.dynamic_capacity == 0:
            return False, False
        evicted = len(self.dynamic) == self.dynamic_capacity
        if evicted:
            self.dynamic.popitem(last=False)
        self.dynamic[expert] = None
        return False, evicted


def _workload_id(path: Path) -> str:
    name = path.name
    suffix = ".trace.jsonl"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return path.stem


def discover_trace_paths(
    *,
    traces: Iterable[str | Path] = (),
    directories: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Resolve explicit traces plus sorted ``*.trace.jsonl`` directory entries."""
    discovered = [Path(path) for path in traces]
    for raw_directory in directories:
        directory = Path(raw_directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"trace directory not found: {directory}")
        discovered.extend(sorted(directory.glob("*.trace.jsonl")))
    if not discovered:
        raise ValueError("at least one trace path is required")
    return tuple(discovered)


def load_placement_traces(paths: Iterable[str | Path]) -> tuple[PlacementTrace, ...]:
    """Load MoEVM JSONL traces and attach stable content identities."""
    loaded: list[PlacementTrace] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        resolved = str(path.resolve())
        if resolved in seen_paths:
            raise ValueError(f"duplicate trace path: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"trace file not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            raise ValueError(f"duplicate trace content: {path}")
        seen_paths.add(resolved)
        seen_hashes.add(digest)
        loaded.append(
            PlacementTrace(
                workload_id=_workload_id(path),
                source=path.as_posix(),
                sha256=digest,
                steps=tuple(read_trace(path)),
            )
        )
    if not loaded:
        raise ValueError("trace collection cannot be empty")
    return tuple(loaded)


def _ordered_steps(trace: PlacementTrace) -> tuple[RoutingStep, ...]:
    if not trace.workload_id.strip():
        raise ValueError("workload_id must be non-empty")
    if not trace.steps:
        raise ValueError(f"trace is empty: {trace.source}")
    ordered = tuple(
        sorted(trace.steps, key=lambda step: (step.token_index, step.layer_index))
    )
    tokens = tuple(sorted({step.token_index for step in ordered}))
    layers = tuple(sorted({step.layer_index for step in ordered}))
    if tokens != tuple(range(tokens[0], tokens[0] + len(tokens))):
        raise ValueError(f"token indices must be contiguous: {trace.source}")
    if layers != tuple(range(layers[0], layers[0] + len(layers))):
        raise ValueError(f"layer indices must be contiguous: {trace.source}")
    addresses = {(step.token_index, step.layer_index) for step in ordered}
    if len(addresses) != len(ordered):
        raise ValueError(f"duplicate token/layer steps: {trace.source}")
    expected = {(token, layer) for token in tokens for layer in layers}
    if addresses != expected:
        raise ValueError(f"incomplete token/layer grid: {trace.source}")
    top_k = len(ordered[0].experts)
    if any(len(step.experts) != top_k for step in ordered):
        raise ValueError(f"inconsistent top-k: {trace.source}")
    return ordered


def _validate_corpus(
    traces: Sequence[PlacementTrace],
) -> tuple[tuple[int, ...], int]:
    if not traces:
        raise ValueError("trace collection cannot be empty")
    expected_layers: tuple[int, ...] | None = None
    expected_top_k: int | None = None
    sources: set[str] = set()
    hashes: set[str] = set()
    for trace in traces:
        if trace.source in sources:
            raise ValueError(f"duplicate trace source: {trace.source}")
        if trace.sha256 in hashes:
            raise ValueError(f"duplicate trace content: {trace.source}")
        sources.add(trace.source)
        hashes.add(trace.sha256)
        ordered = _ordered_steps(trace)
        layers = tuple(sorted({step.layer_index for step in ordered}))
        top_k = len(ordered[0].experts)
        if expected_layers is None:
            expected_layers = layers
            expected_top_k = top_k
        elif layers != expected_layers or top_k != expected_top_k:
            raise ValueError("all traces must use the same layers and top-k")
    assert expected_layers is not None
    assert expected_top_k is not None
    return expected_layers, expected_top_k


def _assert_disjoint(
    train: Sequence[PlacementTrace], test: Sequence[PlacementTrace]
) -> None:
    train_sources = {trace.source for trace in train}
    test_sources = {trace.source for trace in test}
    source_overlap = train_sources.intersection(test_sources)
    if source_overlap:
        raise ValueError(
            f"training/test source overlap would leak test data: {min(source_overlap)}"
        )
    train_hashes = {trace.sha256 for trace in train}
    test_hashes = {trace.sha256 for trace in test}
    if train_hashes.intersection(test_hashes):
        raise ValueError("training/test content overlap would leak test data")


def _addressed_expert_sets(
    traces: Sequence[PlacementTrace],
) -> dict[tuple[str, int, int], set[tuple[int, ...]]]:
    addressed: dict[tuple[str, int, int], set[tuple[int, ...]]] = {}
    for trace in traces:
        for step in _ordered_steps(trace):
            address = (trace.workload_id, step.token_index, step.layer_index)
            addressed.setdefault(address, set()).add(tuple(sorted(step.experts)))
    return addressed


def _split_audit(
    train: Sequence[PlacementTrace], test: Sequence[PlacementTrace]
) -> dict[str, object]:
    train_workloads = {trace.workload_id for trace in train}
    test_workloads = {trace.workload_id for trace in test}
    shared_workloads = train_workloads.intersection(test_workloads)
    train_addresses = _addressed_expert_sets(train)
    test_addresses = _addressed_expert_sets(test)
    shared_addresses = set(train_addresses).intersection(test_addresses)
    exact_matches = sum(
        bool(train_addresses[address].intersection(test_addresses[address]))
        for address in shared_addresses
    )
    return {
        "train_workload_ids": sorted(train_workloads),
        "test_workload_ids": sorted(test_workloads),
        "shared_workload_ids": sorted(shared_workloads),
        "workload_holdout": not shared_workloads,
        "train_unique_step_addresses": len(train_addresses),
        "test_unique_step_addresses": len(test_addresses),
        "shared_same_address_steps": len(shared_addresses),
        "shared_same_address_fraction_of_test": (
            0.0 if not test_addresses else len(shared_addresses) / len(test_addresses)
        ),
        "shared_same_address_exact_expert_set_matches": exact_matches,
    }


def _resolve_capacities(
    layers: Sequence[int], capacity_per_layer: int | Mapping[int, int]
) -> dict[int, int]:
    if isinstance(capacity_per_layer, bool):
        raise ValueError("capacity_per_layer must contain integers")
    if isinstance(capacity_per_layer, int):
        capacities = {layer: capacity_per_layer for layer in layers}
    else:
        capacities = dict(capacity_per_layer)
        missing = set(layers).difference(capacities)
        extra = set(capacities).difference(layers)
        if missing or extra:
            raise ValueError(
                "per-layer capacities must match trace layers exactly; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
    if any(
        isinstance(capacity, bool) or not isinstance(capacity, int)
        for capacity in capacities.values()
    ):
        raise ValueError("capacity_per_layer must contain integers")
    if any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("capacity_per_layer cannot be negative")
    return capacities


def _normalize_policies(policies: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(policies)
    if not normalized:
        raise ValueError("at least one placement policy is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("placement policies cannot be repeated")
    unknown = set(normalized).difference(PLACEMENT_POLICIES)
    if unknown:
        raise ValueError(f"unknown placement policy: {min(unknown)}")
    return normalized


def _training_counts(
    train: Sequence[PlacementTrace], layers: Sequence[int]
) -> dict[int, Counter[int]]:
    counts = {layer: Counter() for layer in layers}
    for trace in train:
        for step in _ordered_steps(trace):
            counts[step.layer_index].update(step.experts)
    return counts


def _select_hotsets(
    counts: Mapping[int, Counter[int]], slots: Mapping[int, int]
) -> dict[int, tuple[int, ...]]:
    return {
        layer: tuple(
            expert
            for expert, _count in sorted(
                counts[layer].items(), key=lambda item: (-item[1], item[0])
            )[: slots[layer]]
        )
        for layer in sorted(counts)
    }


def _hotset_digest(hotsets: Mapping[int, Sequence[int]]) -> str:
    canonical = json.dumps(
        {str(layer): list(hotsets[layer]) for layer in sorted(hotsets)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _manifest(trace: PlacementTrace) -> dict[str, object]:
    ordered = _ordered_steps(trace)
    return {
        "workload_id": trace.workload_id,
        "source": trace.source,
        "sha256": trace.sha256,
        "tokens": len({step.token_index for step in ordered}),
        "routing_steps": len(ordered),
        "expert_accesses": sum(len(step.experts) for step in ordered),
    }


def _metric_payload(
    counts: _Counts,
    *,
    tokens: int,
    expert_bytes: int,
    preload_bytes: int,
) -> dict[str, int | float]:
    demand_bytes = counts.misses * expert_bytes
    total_bytes = preload_bytes + demand_bytes
    return {
        "tokens": tokens,
        "accesses": counts.accesses,
        "hits": counts.hits,
        "misses": counts.misses,
        "evictions": counts.evictions,
        "hit_rate": 0.0 if counts.accesses == 0 else counts.hits / counts.accesses,
        "demand_bytes_loaded": demand_bytes,
        "preload_bytes": preload_bytes,
        "total_bytes_loaded": total_bytes,
        "demand_bytes_per_token": 0.0 if tokens == 0 else demand_bytes / tokens,
        "total_bytes_per_token": 0.0 if tokens == 0 else total_bytes / tokens,
    }


def _new_policy_state(
    policy: str,
    *,
    layers: Sequence[int],
    capacities: Mapping[int, int],
    hotsets: Mapping[int, Sequence[int]],
    protected_hot: int,
):
    if policy == COLD_LRU:
        return {layer: _ColdLru(capacities[layer]) for layer in layers}
    if policy == STATIC_HOT:
        return {layer: _StaticHot(frozenset(hotsets[layer])) for layer in layers}
    if policy == HYBRID_HOT_LRU:
        return {
            layer: _HybridHotLru(
                frozenset(hotsets[layer]),
                capacities[layer] - protected_hot,
            )
            for layer in layers
        }
    raise ValueError(f"unknown placement policy: {policy}")


def _evaluate_trace(
    trace: PlacementTrace,
    *,
    policy: str,
    capacities: Mapping[int, int],
    hotsets: Mapping[int, Sequence[int]],
    protected_hot: int,
    expert_bytes: int,
) -> dict[str, object]:
    ordered = _ordered_steps(trace)
    layers = tuple(sorted(capacities))
    states = _new_policy_state(
        policy,
        layers=layers,
        capacities=capacities,
        hotsets=hotsets,
        protected_hot=protected_hot,
    )
    token_counts: dict[int, _Counts] = {}
    layer_counts = {layer: _Counts() for layer in layers}
    total = _Counts()
    for step in ordered:
        per_token = token_counts.setdefault(step.token_index, _Counts())
        for expert in sorted(step.experts):
            hit, evicted = states[step.layer_index].access(expert)
            total.add(hit=hit, evicted=evicted)
            per_token.add(hit=hit, evicted=evicted)
            layer_counts[step.layer_index].add(hit=hit, evicted=evicted)

    preload_by_layer = {layer: len(hotsets[layer]) * expert_bytes for layer in layers}
    preload_bytes = sum(preload_by_layer.values())
    per_token_payload = []
    for token, counts in sorted(token_counts.items()):
        payload = _metric_payload(
            counts,
            tokens=1,
            expert_bytes=expert_bytes,
            preload_bytes=0,
        )
        payload["token_index"] = token
        per_token_payload.append(payload)
    per_layer_payload = []
    token_total = len(token_counts)
    for layer in layers:
        if policy == HYBRID_HOT_LRU:
            reserved_hot_slots = protected_hot
            dynamic_lru_slots = capacities[layer] - protected_hot
        elif policy == STATIC_HOT:
            reserved_hot_slots = capacities[layer]
            dynamic_lru_slots = 0
        else:
            reserved_hot_slots = 0
            dynamic_lru_slots = capacities[layer]
        layer_payload = _metric_payload(
            layer_counts[layer],
            tokens=token_total,
            expert_bytes=expert_bytes,
            preload_bytes=preload_by_layer[layer],
        )
        layer_payload.update(
            {
                "layer": layer,
                "capacity": capacities[layer],
                "resident_hot_slots": len(hotsets[layer]),
                "reserved_hot_slots": reserved_hot_slots,
                "unused_reserved_hot_slots": (reserved_hot_slots - len(hotsets[layer])),
                "dynamic_lru_slots": dynamic_lru_slots,
            }
        )
        per_layer_payload.append(layer_payload)
    return {
        "workload_id": trace.workload_id,
        "source": trace.source,
        "sha256": trace.sha256,
        "metrics": _metric_payload(
            total,
            tokens=token_total,
            expert_bytes=expert_bytes,
            preload_bytes=preload_bytes,
        ),
        "per_token": per_token_payload,
        "per_layer": per_layer_payload,
    }


def _aggregate_trace_results(
    trace_results: Sequence[Mapping[str, object]], *, expert_bytes: int
) -> dict[str, int | float]:
    counts = _Counts()
    tokens = 0
    preload_bytes = 0
    for result in trace_results:
        metrics = result["metrics"]
        assert isinstance(metrics, Mapping)
        tokens += int(metrics["tokens"])
        counts.accesses += int(metrics["accesses"])
        counts.hits += int(metrics["hits"])
        counts.misses += int(metrics["misses"])
        counts.evictions += int(metrics["evictions"])
        preload_bytes += int(metrics["preload_bytes"])
    return _metric_payload(
        counts,
        tokens=tokens,
        expert_bytes=expert_bytes,
        preload_bytes=preload_bytes,
    )


def _fit_and_evaluate(
    train: Sequence[PlacementTrace],
    test: Sequence[PlacementTrace],
    *,
    capacities: Mapping[int, int],
    protected_hot: int,
    expert_bytes: int,
    policies: Sequence[str],
) -> dict[str, object]:
    layers = tuple(sorted(capacities))
    counts = _training_counts(train, layers)
    static_hotsets = _select_hotsets(counts, capacities)
    hybrid_slots = {layer: protected_hot for layer in layers}
    hybrid_hotsets = _select_hotsets(counts, hybrid_slots)
    empty_hotsets = {layer: () for layer in layers}
    placements = {
        COLD_LRU: empty_hotsets,
        STATIC_HOT: static_hotsets,
        HYBRID_HOT_LRU: hybrid_hotsets,
    }
    evaluations: dict[str, object] = {}
    for policy in policies:
        policy_hotsets = placements[policy]
        trace_results = [
            _evaluate_trace(
                trace,
                policy=policy,
                capacities=capacities,
                hotsets=policy_hotsets,
                protected_hot=protected_hot,
                expert_bytes=expert_bytes,
            )
            for trace in test
        ]
        evaluations[policy] = {
            "placement": {
                "trained_on_test": False,
                "hotsets": {
                    str(layer): list(policy_hotsets[layer]) for layer in layers
                },
                "actual_hotset_slots": {
                    str(layer): len(policy_hotsets[layer]) for layer in layers
                },
                "reserved_hot_slots": {
                    str(layer): (
                        protected_hot
                        if policy == HYBRID_HOT_LRU
                        else capacities[layer]
                        if policy == STATIC_HOT
                        else 0
                    )
                    for layer in layers
                },
                "dynamic_lru_slots": {
                    str(layer): (
                        capacities[layer] - protected_hot
                        if policy == HYBRID_HOT_LRU
                        else capacities[layer]
                        if policy == COLD_LRU
                        else 0
                    )
                    for layer in layers
                },
                "sha256": _hotset_digest(policy_hotsets),
            },
            "aggregate": _aggregate_trace_results(
                trace_results, expert_bytes=expert_bytes
            ),
            "per_trace": trace_results,
        }
    return {
        "training": {
            "expert_accesses": sum(sum(layer.values()) for layer in counts.values()),
            "fit_input_sha256": [trace.sha256 for trace in train],
        },
        "evaluations": evaluations,
    }


def analyze_placement_train_test(
    train: Sequence[PlacementTrace],
    test: Sequence[PlacementTrace],
    *,
    capacity_per_layer: int | Mapping[int, int],
    protected_hot: int,
    expert_bytes: int,
    policies: Iterable[str] = PLACEMENT_POLICIES,
) -> dict[str, object]:
    """Fit placement on ``train`` once, then evaluate immutable policies on test."""
    train_layers, train_top_k = _validate_corpus(train)
    test_layers, test_top_k = _validate_corpus(test)
    if train_layers != test_layers:
        raise ValueError("training and test traces must use the same layers")
    if train_top_k != test_top_k:
        raise ValueError(
            "training and test traces must use the same top-k: "
            f"{train_top_k} != {test_top_k}"
        )
    _assert_disjoint(train, test)
    normalized_policies = _normalize_policies(policies)
    capacities = _resolve_capacities(train_layers, capacity_per_layer)
    if isinstance(protected_hot, bool) or not isinstance(protected_hot, int):
        raise ValueError("protected_hot must be an integer")
    if protected_hot < 0 or any(
        protected_hot > capacity for capacity in capacities.values()
    ):
        raise ValueError("protected_hot must fit every per-layer capacity")
    if isinstance(expert_bytes, bool) or not isinstance(expert_bytes, int):
        raise ValueError("expert_bytes must be an integer")
    if expert_bytes <= 0:
        raise ValueError("expert_bytes must be positive")

    core = _fit_and_evaluate(
        train,
        test,
        capacities=capacities,
        protected_hot=protected_hot,
        expert_bytes=expert_bytes,
        policies=normalized_policies,
    )
    return {
        "schema_version": 2,
        "label": "cross-seed placement trace replay; shared workloads audited",
        "protocol": "train-test",
        "fit_controls": {
            "fit_uses_training_only": True,
            "test_updates_placement": False,
            "cache_reset_per_test_trace": True,
            "source_and_sha256_overlap_rejected": True,
        },
        "split_audit": _split_audit(train, test),
        "parameters": {
            "expert_bytes": expert_bytes,
            "top_k": train_top_k,
            "capacity_per_layer": {
                str(layer): capacities[layer] for layer in train_layers
            },
            "hybrid_protected_hot_per_layer": protected_hot,
            "hybrid_dynamic_lru_per_layer": {
                str(layer): capacities[layer] - protected_hot for layer in train_layers
            },
            "intra_step_access_order": "expert_id_ascending",
            "static_miss_scratch_counted_in_capacity": False,
            "prefill_or_request_batching_modeled": False,
            "hyperparameters_selected_by_nested_validation": False,
            "policies": list(normalized_policies),
        },
        "train_manifest": [_manifest(trace) for trace in train],
        "test_manifest": [_manifest(trace) for trace in test],
        **core,
    }


def analyze_placement_leave_one_workload_out(
    train: Sequence[PlacementTrace],
    test: Sequence[PlacementTrace],
    *,
    capacity_per_layer: int | Mapping[int, int],
    protected_hot: int,
    expert_bytes: int,
    policies: Iterable[str] = PLACEMENT_POLICIES,
) -> dict[str, object]:
    """Evaluate each test workload after excluding it entirely from fitting."""
    train_layers, train_top_k = _validate_corpus(train)
    test_layers, test_top_k = _validate_corpus(test)
    if train_layers != test_layers:
        raise ValueError("training and test traces must use the same layers")
    if train_top_k != test_top_k:
        raise ValueError(
            "training and test traces must use the same top-k: "
            f"{train_top_k} != {test_top_k}"
        )
    _assert_disjoint(train, test)
    normalized_policies = _normalize_policies(policies)
    capacities = _resolve_capacities(train_layers, capacity_per_layer)
    if isinstance(protected_hot, bool) or not isinstance(protected_hot, int):
        raise ValueError("protected_hot must be an integer")
    if protected_hot < 0 or any(
        protected_hot > capacity for capacity in capacities.values()
    ):
        raise ValueError("protected_hot must fit every per-layer capacity")
    if isinstance(expert_bytes, bool) or not isinstance(expert_bytes, int):
        raise ValueError("expert_bytes must be an integer")
    if expert_bytes <= 0:
        raise ValueError("expert_bytes must be positive")

    folds: list[dict[str, object]] = []
    all_policy_results: dict[str, list[Mapping[str, object]]] = {
        policy: [] for policy in normalized_policies
    }
    for held_out in sorted({trace.workload_id for trace in test}):
        fold_train = tuple(trace for trace in train if trace.workload_id != held_out)
        fold_test = tuple(trace for trace in test if trace.workload_id == held_out)
        if not fold_train:
            raise ValueError(f"no training traces remain after holding out {held_out}")
        if any(trace.workload_id == held_out for trace in fold_train):
            raise AssertionError("held-out workload leaked into training")
        core = _fit_and_evaluate(
            fold_train,
            fold_test,
            capacities=capacities,
            protected_hot=protected_hot,
            expert_bytes=expert_bytes,
            policies=normalized_policies,
        )
        evaluations = core["evaluations"]
        assert isinstance(evaluations, Mapping)
        for policy in normalized_policies:
            evaluation = evaluations[policy]
            assert isinstance(evaluation, Mapping)
            per_trace = evaluation["per_trace"]
            assert isinstance(per_trace, list)
            all_policy_results[policy].extend(per_trace)
        folds.append(
            {
                "held_out_workload": held_out,
                "split_audit": _split_audit(fold_train, fold_test),
                "train_manifest": [_manifest(trace) for trace in fold_train],
                "test_manifest": [_manifest(trace) for trace in fold_test],
                **core,
            }
        )

    aggregate = {
        policy: _aggregate_trace_results(
            all_policy_results[policy], expert_bytes=expert_bytes
        )
        for policy in normalized_policies
    }
    matrix = [
        {
            "held_out_workload": fold["held_out_workload"],
            "policies": {
                policy: fold["evaluations"][policy]["aggregate"]
                for policy in normalized_policies
            },
        }
        for fold in folds
    ]
    return {
        "schema_version": 2,
        "label": "strict leave-one-workload-out placement trace replay",
        "protocol": "leave-one-workload-out",
        "fit_controls": {
            "fit_uses_training_only": True,
            "held_out_workload_excluded_from_fit": True,
            "test_updates_placement": False,
            "cache_reset_per_test_trace": True,
            "source_and_sha256_overlap_rejected": True,
        },
        "split_audit": {
            "strict_workload_holdout_by_fold": True,
            "folds": len(folds),
            "folds_with_shared_workloads": sum(
                bool(fold["split_audit"]["shared_workload_ids"]) for fold in folds
            ),
            "folds_with_shared_same_address_steps": sum(
                bool(fold["split_audit"]["shared_same_address_steps"]) for fold in folds
            ),
        },
        "parameters": {
            "expert_bytes": expert_bytes,
            "top_k": train_top_k,
            "capacity_per_layer": {
                str(layer): capacities[layer] for layer in train_layers
            },
            "hybrid_protected_hot_per_layer": protected_hot,
            "hybrid_dynamic_lru_per_layer": {
                str(layer): capacities[layer] - protected_hot for layer in train_layers
            },
            "intra_step_access_order": "expert_id_ascending",
            "static_miss_scratch_counted_in_capacity": False,
            "prefill_or_request_batching_modeled": False,
            "hyperparameters_selected_by_nested_validation": False,
            "policies": list(normalized_policies),
        },
        "source_train_manifest": [_manifest(trace) for trace in train],
        "source_test_manifest": [_manifest(trace) for trace in test],
        "folds": folds,
        "matrix": matrix,
        "aggregate_across_folds": aggregate,
    }


def placement_analysis_console(report: Mapping[str, object]) -> str:
    parameters = report["parameters"]
    assert isinstance(parameters, Mapping)
    policies = parameters["policies"]
    assert isinstance(policies, list)
    if report["protocol"] == "train-test":
        evaluations = report["evaluations"]
        assert isinstance(evaluations, Mapping)
        aggregates = {policy: evaluations[policy]["aggregate"] for policy in policies}
    else:
        aggregates = report["aggregate_across_folds"]
        assert isinstance(aggregates, Mapping)
    lines = [
        "MoEVM Lab — offline placement analysis",
        "========================================",
        "TRACE REPLAY ONLY: no runtime timing is measured.",
        f"Protocol: {report['protocol']}",
    ]
    split_audit = report["split_audit"]
    assert isinstance(split_audit, Mapping)
    if report["protocol"] == "train-test":
        lines.append(
            "Cross-seed audit: "
            f"{len(split_audit['shared_workload_ids'])} shared workloads; "
            f"{split_audit['shared_same_address_steps']:,}/"
            f"{split_audit['test_unique_step_addresses']:,} test step addresses "
            "also occur in training."
        )
    else:
        lines.append(
            "Strict workload holdout: "
            f"{split_audit['folds']} folds; "
            f"{split_audit['folds_with_shared_workloads']} with shared workloads."
        )
    for policy in policies:
        metrics = aggregates[policy]
        lines.append(
            f"{policy}: hit {metrics['hit_rate'] * 100:.2f}%; "
            f"misses {metrics['misses']:,}; evictions {metrics['evictions']:,}; "
            f"demand/total bytes per token "
            f"{metrics['demand_bytes_per_token']:,.1f}/"
            f"{metrics['total_bytes_per_token']:,.1f}"
        )
    return "\n".join(lines)


def write_placement_analysis(output: str | Path, report: Mapping[str, object]) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return destination
