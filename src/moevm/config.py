from __future__ import annotations

import dataclasses
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_MIB = 1024 * 1024


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    layers: int
    experts_per_layer: int
    top_k: int
    expert_size_mib: float
    compute_ms_per_layer: float

    @property
    def expert_size_bytes(self) -> int:
        return max(1, int(self.expert_size_mib * _MIB))

    def validate(self) -> None:
        _require_string("model.name", self.name)
        if _require_int("model.layers", self.layers) <= 0:
            raise ValueError("model.layers must be positive")
        if _require_int("model.experts_per_layer", self.experts_per_layer) <= 0:
            raise ValueError("model.experts_per_layer must be positive")
        if not 0 < _require_int("model.top_k", self.top_k) <= self.experts_per_layer:
            raise ValueError("model.top_k must be between 1 and experts_per_layer")
        if _require_finite_number("model.expert_size_mib", self.expert_size_mib) <= 0:
            raise ValueError("model.expert_size_mib must be positive")
        if (
            _require_finite_number(
                "model.compute_ms_per_layer", self.compute_ms_per_layer
            )
            < 0
        ):
            raise ValueError("model.compute_ms_per_layer cannot be negative")


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    vram_cache_mib: float
    ram_cache_mib: float
    ram_to_vram_gbps: float
    nvme_to_ram_gbps: float
    ram_latency_us: float
    nvme_latency_us: float
    overlap_efficiency: float
    prefetch_vram_fraction: float

    @property
    def vram_cache_bytes(self) -> int:
        return max(0, int(self.vram_cache_mib * _MIB))

    @property
    def ram_cache_bytes(self) -> int:
        return max(0, int(self.ram_cache_mib * _MIB))

    def validate(self) -> None:
        vram_cache_mib = _require_finite_number(
            "hardware.vram_cache_mib", self.vram_cache_mib
        )
        ram_cache_mib = _require_finite_number(
            "hardware.ram_cache_mib", self.ram_cache_mib
        )
        if vram_cache_mib < 0 or ram_cache_mib < 0:
            raise ValueError("cache sizes cannot be negative")
        ram_to_vram_gbps = _require_finite_number(
            "hardware.ram_to_vram_gbps", self.ram_to_vram_gbps
        )
        nvme_to_ram_gbps = _require_finite_number(
            "hardware.nvme_to_ram_gbps", self.nvme_to_ram_gbps
        )
        if ram_to_vram_gbps <= 0 or nvme_to_ram_gbps <= 0:
            raise ValueError("bandwidth values must be positive")
        ram_latency_us = _require_finite_number(
            "hardware.ram_latency_us", self.ram_latency_us
        )
        nvme_latency_us = _require_finite_number(
            "hardware.nvme_latency_us", self.nvme_latency_us
        )
        if ram_latency_us < 0 or nvme_latency_us < 0:
            raise ValueError("latency values cannot be negative")
        overlap_efficiency = _require_finite_number(
            "hardware.overlap_efficiency", self.overlap_efficiency
        )
        if not 0.0 <= overlap_efficiency <= 1.0:
            raise ValueError("hardware.overlap_efficiency must be in [0, 1]")
        prefetch_vram_fraction = _require_finite_number(
            "hardware.prefetch_vram_fraction", self.prefetch_vram_fraction
        )
        if not 0.0 <= prefetch_vram_fraction < 1.0:
            raise ValueError("hardware.prefetch_vram_fraction must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TraceConfig:
    tokens: int
    domains: int
    domain_switch_probability: float
    hotset_multiplier: float
    temporal_reuse_probability: float
    random_expert_probability: float
    seed: int

    def validate(self) -> None:
        if _require_int("trace.tokens", self.tokens) <= 0:
            raise ValueError("trace.tokens must be positive")
        if _require_int("trace.domains", self.domains) <= 0:
            raise ValueError("trace.domains must be positive")
        if (
            _require_finite_number("trace.hotset_multiplier", self.hotset_multiplier)
            < 1
        ):
            raise ValueError("trace.hotset_multiplier must be at least 1")
        for name in (
            "domain_switch_probability",
            "temporal_reuse_probability",
            "random_expert_probability",
        ):
            value = _require_finite_number(f"trace.{name}", getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"trace.{name} must be in [0, 1]")
        _require_int("trace.seed", self.seed)


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    enabled: bool
    prefetch_count: int
    frequency_weight: float
    temporal_weight: float
    cross_layer_weight: float
    max_targets_per_source: int
    min_relative_confidence: float
    deadline_aware: bool

    def validate(self) -> None:
        _require_bool("predictor.enabled", self.enabled)
        if _require_int("predictor.prefetch_count", self.prefetch_count) < 0:
            raise ValueError("predictor.prefetch_count cannot be negative")
        weights = (
            _require_finite_number("predictor.frequency_weight", self.frequency_weight),
            _require_finite_number("predictor.temporal_weight", self.temporal_weight),
            _require_finite_number(
                "predictor.cross_layer_weight", self.cross_layer_weight
            ),
        )
        if min(weights) < 0:
            raise ValueError("predictor weights cannot be negative")
        if (
            _require_int(
                "predictor.max_targets_per_source", self.max_targets_per_source
            )
            <= 0
        ):
            raise ValueError("predictor.max_targets_per_source must be positive")
        min_relative_confidence = _require_finite_number(
            "predictor.min_relative_confidence", self.min_relative_confidence
        )
        if not 0.0 <= min_relative_confidence <= 1.0:
            raise ValueError("predictor.min_relative_confidence must be in [0, 1]")
        _require_bool("predictor.deadline_aware", self.deadline_aware)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    hardware: HardwareConfig
    trace: TraceConfig
    predictor: PredictorConfig

    def validate(self) -> None:
        self.model.validate()
        self.hardware.validate()
        self.trace.validate()
        self.predictor.validate()
        if self.predictor.enabled and self.predictor.prefetch_count:
            speculative_bytes = int(
                self.hardware.vram_cache_bytes * self.hardware.prefetch_vram_fraction
            )
            if speculative_bytes < self.model.expert_size_bytes:
                raise ValueError(
                    "the speculative VRAM partition must fit at least one expert "
                    "when prefetching is enabled"
                )

    def with_tokens(self, tokens: int | None) -> ExperimentConfig:
        if tokens is None:
            return self
        updated = replace(self, trace=replace(self.trace, tokens=tokens))
        updated.validate()
        return updated

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid [{name}] section")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    try:
        config = ExperimentConfig(
            model=ModelConfig(**_section(data, "model")),
            hardware=HardwareConfig(**_section(data, "hardware")),
            trace=TraceConfig(**_section(data, "trace")),
            predictor=PredictorConfig(**_section(data, "predictor")),
        )
    except TypeError as exc:
        raise ValueError(
            f"invalid configuration schema in {config_path}: {exc}"
        ) from exc

    config.validate()
    return config
