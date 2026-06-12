"""Seeded Latent Trajectory Aggregation for DeepSeek-MoE."""

from .aggregator import BaseAnchoredResidualAggregator
from .modeling_trajectory import TrajectoryCausalLMOutput, TrajectoryEnsembleForCausalLM
from .router_noise import SeedRouterNoise, patch_deepseek_moe_gates

__all__ = [
    "BaseAnchoredResidualAggregator",
    "SeedRouterNoise",
    "TrajectoryCausalLMOutput",
    "TrajectoryEnsembleForCausalLM",
    "patch_deepseek_moe_gates",
]

