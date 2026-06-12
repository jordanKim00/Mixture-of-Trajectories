from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class AggregatorStats:
    """Small CPU-friendly summary of trajectory attention."""

    alpha_mean: torch.Tensor
    alpha_std: torch.Tensor
    alpha_entropy_mean: float
    alpha_max_mean: float
    null_alpha_mean: Optional[float]
    alt_alpha_mass_mean: float
    residual_scale: float
    residual_norm_ratio: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "alpha_mean": self.alpha_mean.tolist(),
            "alpha_std": self.alpha_std.tolist(),
            "alpha_entropy_mean": self.alpha_entropy_mean,
            "alpha_max_mean": self.alpha_max_mean,
            "null_alpha_mean": self.null_alpha_mean,
            "alt_alpha_mass_mean": self.alt_alpha_mass_mean,
            "residual_scale": self.residual_scale,
            "residual_norm_ratio": self.residual_norm_ratio,
        }


class BaseAnchoredResidualAggregator(nn.Module):
    """Fuse final trajectory hidden states without moving the LM head space.

    Input shape is ``(batch, trajectories, seq, hidden)``. Trajectory 0 is the
    frozen base route. Trajectories 1..N-1 are alternative seeded routes.
    The module returns a pre-final-norm hidden state with the same shape as the
    base route: ``(batch, seq, hidden)``.
    """

    def __init__(
        self,
        hidden_size: int,
        agg_dim: int = 256,
        residual_scale_init: float = 0.01,
        residual_scale_max: float = 0.25,
        include_null_candidate: bool = True,
        value_mode: str = "delta",
        relative_keys: bool = False,
    ) -> None:
        super().__init__()
        if agg_dim <= 0:
            raise ValueError("agg_dim must be positive")
        if not 0.0 < residual_scale_init < residual_scale_max:
            raise ValueError("residual_scale_init must be in (0, residual_scale_max)")
        if value_mode not in ("delta", "absolute"):
            raise ValueError("value_mode must be one of {'delta', 'absolute'}")

        self.hidden_size = int(hidden_size)
        self.agg_dim = int(agg_dim)
        self.residual_scale_max = float(residual_scale_max)
        self.include_null_candidate = bool(include_null_candidate)
        self.value_mode = value_mode
        self.relative_keys = bool(relative_keys)
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, agg_dim, bias=False)
        self.global_q_proj = nn.Linear(hidden_size, agg_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, agg_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, agg_dim, bias=False)
        self.out_proj = nn.Linear(agg_dim, hidden_size, bias=True)
        if self.include_null_candidate:
            self.null_key = nn.Parameter(torch.zeros(agg_dim))
        else:
            self.register_parameter("null_key", None)
        raw_init = math.atanh(float(residual_scale_init) / self.residual_scale_max)
        self.raw_residual_scale = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # Exact identity initialization: delta is zero. The output projection
        # moves first; q/k/v and router noise receive signal once it is nonzero.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def residual_scale_value(self) -> torch.Tensor:
        return self.residual_scale_max * torch.tanh(self.raw_residual_scale)

    def forward(
        self,
        hidden_by_traj: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        return_alpha: bool = False,
    ) -> Tuple[torch.Tensor, Optional[AggregatorStats]] | Tuple[torch.Tensor, Optional[AggregatorStats], torch.Tensor]:
        if hidden_by_traj.ndim != 4:
            raise ValueError(
                "hidden_by_traj must have shape (batch, trajectories, seq, hidden)"
            )
        batch, num_trajectories, seq_len, hidden = hidden_by_traj.shape
        if hidden != self.hidden_size:
            raise ValueError(f"expected hidden={self.hidden_size}, got {hidden}")
        if num_trajectories < 2:
            raise ValueError("at least one base and one noisy trajectory are required")

        base_hidden = hidden_by_traj[:, 0]  # (B, S, D)
        alt_hidden = hidden_by_traj[:, 1:]  # (B, P, S, D)
        compute_dtype = self.q_proj.weight.dtype

        base_normed = self.norm(base_hidden.to(compute_dtype))
        q = self.q_proj(base_normed)  # (B, S, A)
        if attention_mask is not None:
            mask = attention_mask.to(device=base_hidden.device, dtype=compute_dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            global_hidden = (base_normed * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            global_hidden = base_normed.mean(dim=1)
        q = q + self.global_q_proj(global_hidden).unsqueeze(1)
        alt_normed = self.norm(alt_hidden.to(compute_dtype))
        k = self.k_proj(alt_normed)  # (B, P, S, A)
        if self.relative_keys and k.shape[1] > 1:
            # Comparative judge: score each candidate against the others
            # instead of absolutely, so a direction shared by every noisy
            # trajectory carries no selection signal.
            k = k - k.mean(dim=1, keepdim=True)
        if self.value_mode == "delta":
            value_input = alt_normed - base_normed.unsqueeze(1)
        else:
            value_input = alt_normed
        v = self.v_proj(value_input)  # (B, P, S, A)

        scores = (q.unsqueeze(1) * k).sum(dim=-1) / math.sqrt(self.agg_dim)
        if self.include_null_candidate:
            null_key = self.null_key.to(dtype=compute_dtype, device=q.device)
            null_scores = (q * null_key.view(1, 1, -1)).sum(dim=-1, keepdim=True)
            null_scores = null_scores.transpose(1, 2) / math.sqrt(self.agg_dim)
            scores = torch.cat([scores, null_scores], dim=1)
        alpha = torch.softmax(scores, dim=1)  # (B, P, S)
        alt_alpha = alpha[:, : alt_hidden.shape[1]]
        context = (alt_alpha.unsqueeze(-1) * v).sum(dim=1)  # (B, S, A)
        delta = self.out_proj(context)
        scale = self.residual_scale_value().to(delta.dtype)
        residual = scale * delta
        fused = base_hidden + residual.to(base_hidden.dtype)

        stats = None
        if return_stats:
            alpha_detached = alpha.detach().float()
            if attention_mask is not None:
                valid = attention_mask.to(device=alpha.device, dtype=torch.bool)
            else:
                valid = torch.ones(batch, seq_len, device=alpha.device, dtype=torch.bool)
            valid_alpha = valid[:, None, :]
            denom = valid_alpha.float().sum(dim=(0, 2)).clamp_min(1.0)
            alpha_mean = (alpha_detached * valid_alpha).sum(dim=(0, 2)) / denom
            alpha_var = ((alpha_detached - alpha_mean.view(1, -1, 1)).pow(2) * valid_alpha).sum(
                dim=(0, 2)
            ) / denom
            alpha_entropy = -(alpha_detached * alpha_detached.clamp_min(1e-20).log()).sum(dim=1)
            token_denom = valid.float().sum().clamp_min(1.0)
            alpha_entropy_mean = (alpha_entropy * valid).sum() / token_denom
            alpha_max_mean = (alpha_detached.max(dim=1).values * valid).sum() / token_denom
            if self.include_null_candidate:
                null_alpha = alpha_detached[:, -1]
                null_alpha_mean = (null_alpha * valid).sum() / token_denom
                alt_alpha_mass = (alpha_detached[:, :-1].sum(dim=1) * valid).sum() / token_denom
                null_alpha_value: Optional[float] = float(null_alpha_mean.cpu().item())
            else:
                alt_alpha_mass = (alpha_detached.sum(dim=1) * valid).sum() / token_denom
                null_alpha_value = None
            residual_norm = (
                residual.detach().float().norm(dim=-1) * valid
            ).sum() / token_denom
            base_norm = (
                base_hidden.detach().float().norm(dim=-1) * valid
            ).sum() / token_denom
            base_norm = base_norm.clamp_min(1e-12)
            stats = AggregatorStats(
                alpha_mean=alpha_mean.cpu(),
                alpha_std=alpha_var.sqrt().cpu(),
                alpha_entropy_mean=float(alpha_entropy_mean.cpu().item()),
                alpha_max_mean=float(alpha_max_mean.cpu().item()),
                null_alpha_mean=null_alpha_value,
                alt_alpha_mass_mean=float(alt_alpha_mass.cpu().item()),
                residual_scale=float(self.residual_scale_value().detach().float().cpu().item()),
                residual_norm_ratio=float((residual_norm / base_norm).cpu().item()),
            )
        if return_alpha:
            return fused, stats, alpha
        return fused, stats
