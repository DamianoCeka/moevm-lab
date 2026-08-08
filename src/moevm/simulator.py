from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Iterable, Literal

from .cache import HierarchicalExpertCache
from .config import ExperimentConfig
from .predictor import OnlineExpertPredictor
from .trace import SyntheticRoutingTrace
from .types import ExpertKey, RoutingStep

RunMode = Literal["baseline", "prefetch"]


@dataclass(slots=True)
class RunMetrics:
    mode: str
    model_name: str
    tokens: int = 0
    steps: int = 0
    expert_accesses: int = 0
    elapsed_ms: float = 0.0
    compute_ms: float = 0.0
    demand_stall_ms: float = 0.0
    prefetch_stall_ms: float = 0.0
    demand_l1_hits: int = 0
    demand_l2_hits: int = 0
    demand_storage_hits: int = 0
    demand_ram_to_vram_bytes: int = 0
    demand_nvme_to_ram_bytes: int = 0
    prefetch_predictions: int = 0
    prefetch_candidates: int = 0
    prefetch_rejected_deadline: int = 0
    prefetch_l1_hits: int = 0
    prefetch_l2_hits: int = 0
    prefetch_storage_hits: int = 0
    prefetch_loaded: int = 0
    prefetch_useful: int = 0
    prefetch_wasted: int = 0
    prefetch_ram_to_vram_bytes: int = 0
    prefetch_nvme_to_ram_bytes: int = 0
    final_cache: dict[str, int] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return 0.0 if self.elapsed_ms <= 0 else self.tokens / (self.elapsed_ms / 1000.0)

    @property
    def demand_l1_hit_rate(self) -> float:
        return 0.0 if self.expert_accesses == 0 else self.demand_l1_hits / self.expert_accesses

    @property
    def demand_cache_hit_rate(self) -> float:
        hits = self.demand_l1_hits + self.demand_l2_hits
        return 0.0 if self.expert_accesses == 0 else hits / self.expert_accesses

    @property
    def prefetch_precision(self) -> float:
        return 0.0 if self.prefetch_loaded == 0 else self.prefetch_useful / self.prefetch_loaded

    @property
    def total_nvme_to_ram_bytes(self) -> int:
        return self.demand_nvme_to_ram_bytes + self.prefetch_nvme_to_ram_bytes

    @property
    def total_ram_to_vram_bytes(self) -> int:
        return self.demand_ram_to_vram_bytes + self.prefetch_ram_to_vram_bytes

    @property
    def total_stall_ms(self) -> float:
        return self.demand_stall_ms + self.prefetch_stall_ms

    @property
    def nvme_bytes_per_token(self) -> float:
        return 0.0 if self.tokens == 0 else self.total_nvme_to_ram_bytes / self.tokens

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = dataclasses.asdict(self)
        values.update(
            {
                "tokens_per_second": self.tokens_per_second,
                "demand_l1_hit_rate": self.demand_l1_hit_rate,
                "demand_cache_hit_rate": self.demand_cache_hit_rate,
                "prefetch_precision": self.prefetch_precision,
                "total_nvme_to_ram_bytes": self.total_nvme_to_ram_bytes,
                "total_ram_to_vram_bytes": self.total_ram_to_vram_bytes,
                "total_stall_ms": self.total_stall_ms,
                "nvme_bytes_per_token": self.nvme_bytes_per_token,
            }
        )
        return values


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    baseline: RunMetrics
    prefetch: RunMetrics

    @property
    def speedup(self) -> float:
        if self.prefetch.elapsed_ms <= 0:
            return 0.0
        return self.baseline.elapsed_ms / self.prefetch.elapsed_ms

    @property
    def demand_nvme_reduction(self) -> float:
        baseline = self.baseline.demand_nvme_to_ram_bytes
        if baseline <= 0:
            return 0.0
        return 1.0 - (self.prefetch.demand_nvme_to_ram_bytes / baseline)

    @property
    def total_nvme_reduction(self) -> float:
        baseline = self.baseline.total_nvme_to_ram_bytes
        if baseline <= 0:
            return 0.0
        return 1.0 - (self.prefetch.total_nvme_to_ram_bytes / baseline)

    @property
    def demand_stall_reduction(self) -> float:
        baseline = self.baseline.demand_stall_ms
        if baseline <= 0:
            return 0.0
        return 1.0 - (self.prefetch.demand_stall_ms / baseline)

    @property
    def ram_to_vram_traffic_change(self) -> float:
        baseline = self.baseline.total_ram_to_vram_bytes
        if baseline <= 0:
            return 0.0
        return (self.prefetch.total_ram_to_vram_bytes / baseline) - 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "prefetch": self.prefetch.to_dict(),
            "comparison": {
                "speedup": self.speedup,
                "demand_nvme_reduction": self.demand_nvme_reduction,
                "total_nvme_reduction": self.total_nvme_reduction,
                "demand_stall_reduction": self.demand_stall_reduction,
                "ram_to_vram_traffic_change": self.ram_to_vram_traffic_change,
            },
        }


def _validate_trace(config: ExperimentConfig, steps: list[RoutingStep]) -> None:
    if not steps:
        raise ValueError("trace cannot be empty")
    for index, step in enumerate(steps):
        if step.layer_index >= config.model.layers:
            raise ValueError(f"trace step {index} references invalid layer {step.layer_index}")
        if len(step.experts) != config.model.top_k:
            raise ValueError(
                f"trace step {index} has {len(step.experts)} experts; expected {config.model.top_k}"
            )
        if max(step.experts) >= config.model.experts_per_layer:
            raise ValueError(f"trace step {index} references an invalid expert")


def _mark_evictions(
    evicted: Iterable[ExpertKey],
    active_prefetch: set[ExpertKey],
    metrics: RunMetrics,
) -> None:
    for key in evicted:
        if key in active_prefetch:
            active_prefetch.remove(key)
            metrics.prefetch_wasted += 1


def run_experiment(
    config: ExperimentConfig,
    mode: RunMode = "prefetch",
    trace: Iterable[RoutingStep] | None = None,
) -> RunMetrics:
    config.validate()
    steps = list(trace) if trace is not None else list(SyntheticRoutingTrace(config).generate())
    _validate_trace(config, steps)

    use_prefetch = mode == "prefetch" and config.predictor.enabled
    cache = HierarchicalExpertCache(
        config.hardware,
        config.model.expert_size_bytes,
        prefetch_enabled=use_prefetch,
    )
    predictor = OnlineExpertPredictor(config.predictor)
    metrics = RunMetrics(mode=mode, model_name=config.model.name)
    metrics.tokens = len({step.token_index for step in steps})
    metrics.steps = len(steps)

    pending_prefetch: tuple[ExpertKey, ...] = ()
    active_prefetch: set[ExpertKey] = set()
    previous_step: RoutingStep | None = None
    previous_compute_ms = 0.0

    for index, step in enumerate(steps):
        if use_prefetch and pending_prefetch:
            metrics.prefetch_predictions += len(pending_prefetch)
            admitted_prefetch = pending_prefetch
            if config.predictor.deadline_aware:
                budget_ms = previous_compute_ms * config.hardware.overlap_efficiency
                admitted: list[ExpertKey] = []
                for key in pending_prefetch:
                    trial = (*admitted, key)
                    if cache.estimate_transfer_ms(trial) <= budget_ms + 1e-12:
                        admitted.append(key)
                    else:
                        metrics.prefetch_rejected_deadline += 1
                admitted_prefetch = tuple(admitted)

            prefetch = cache.prefetch_many(admitted_prefetch)
            metrics.prefetch_candidates += prefetch.requested
            metrics.prefetch_l1_hits += prefetch.l1_hits
            metrics.prefetch_l2_hits += prefetch.l2_hits
            metrics.prefetch_storage_hits += prefetch.storage_hits
            metrics.prefetch_ram_to_vram_bytes += prefetch.bytes_ram_to_vram
            metrics.prefetch_nvme_to_ram_bytes += prefetch.bytes_nvme_to_ram
            _mark_evictions(prefetch.evicted_from_l1, active_prefetch, metrics)

            for key in prefetch.loaded_to_l1:
                if key not in active_prefetch:
                    active_prefetch.add(key)
                    metrics.prefetch_loaded += 1

            hidden_budget_ms = previous_compute_ms * config.hardware.overlap_efficiency
            metrics.prefetch_stall_ms += max(0.0, prefetch.transfer_ms - hidden_budget_ms)

        required = step.keys
        useful_now = active_prefetch.intersection(required)
        demand = cache.access_many(required)
        metrics.expert_accesses += demand.requested
        metrics.demand_l1_hits += demand.l1_hits
        metrics.demand_l2_hits += demand.l2_hits
        metrics.demand_storage_hits += demand.storage_hits
        metrics.demand_ram_to_vram_bytes += demand.bytes_ram_to_vram
        metrics.demand_nvme_to_ram_bytes += demand.bytes_nvme_to_ram
        metrics.demand_stall_ms += demand.transfer_ms

        for key in useful_now:
            active_prefetch.remove(key)
            metrics.prefetch_useful += 1
        _mark_evictions(demand.evicted_from_l1, active_prefetch, metrics)

        metrics.compute_ms += config.model.compute_ms_per_layer
        previous_compute_ms = config.model.compute_ms_per_layer

        if use_prefetch:
            predictor.observe(step, previous_step)
            if index + 1 < len(steps):
                next_step = steps[index + 1]
                predicted_experts = predictor.predict(
                    target_layer=next_step.layer_index,
                    current_step=step,
                    limit=min(
                        config.predictor.prefetch_count,
                        config.model.experts_per_layer,
                    ),
                )
                pending_prefetch = tuple(
                    ExpertKey(next_step.layer_index, expert) for expert in predicted_experts
                )
            else:
                pending_prefetch = ()
        else:
            pending_prefetch = ()

        previous_step = step

    metrics.prefetch_wasted += len(active_prefetch)
    metrics.final_cache = cache.snapshot()
    metrics.elapsed_ms = metrics.compute_ms + metrics.demand_stall_ms + metrics.prefetch_stall_ms
    return metrics


def compare_experiment(
    config: ExperimentConfig,
    trace: Iterable[RoutingStep] | None = None,
) -> ComparisonResult:
    steps = list(trace) if trace is not None else list(SyntheticRoutingTrace(config).generate())
    baseline = run_experiment(config, mode="baseline", trace=steps)
    prefetch = run_experiment(config, mode="prefetch", trace=steps)
    return ComparisonResult(baseline=baseline, prefetch=prefetch)
