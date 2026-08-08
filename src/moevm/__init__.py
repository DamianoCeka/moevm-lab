"""MoEVM Lab: expert-aware memory virtualization research tools."""

from .config import ExperimentConfig, load_config
from .simulator import ComparisonResult, RunMetrics, compare_experiment, run_experiment

__all__ = [
    "ComparisonResult",
    "ExperimentConfig",
    "RunMetrics",
    "compare_experiment",
    "load_config",
    "run_experiment",
]

__version__ = "0.1.0"
