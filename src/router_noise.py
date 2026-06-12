from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RoutingLayerStats:
    layer_idx: int
    noise_applied: bool
    entropy_by_traj: List[float]
    topk_overlap_with_base: List[float]
    topk_exact_match_with_base: List[float]
    topk_margin_by_traj: List[float]
    topk_logit_margin_by_traj: List[float]
    expert_probs_by_traj: List[List[float]]
    expert_jsd_with_base: List[float]
    seed_scale_by_traj: List[float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "noise_applied": self.noise_applied,
            "entropy_by_traj": self.entropy_by_traj,
            "topk_overlap_with_base": self.topk_overlap_with_base,
            "topk_exact_match_with_base": self.topk_exact_match_with_base,
            "topk_margin_by_traj": self.topk_margin_by_traj,
            "topk_logit_margin_by_traj": self.topk_logit_margin_by_traj,
            "expert_probs_by_traj": self.expert_probs_by_traj,
            "expert_jsd_with_base": self.expert_jsd_with_base,
            "seed_scale_by_traj": self.seed_scale_by_traj,
        }


@dataclass
class RoutingPathStats:
    layer_idx: int
    layer_exact_match_with_base: List[float]
    prefix_exact_match_with_base: List[float]
    new_divergence_rate_from_previous: List[float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "layer_exact_match_with_base": self.layer_exact_match_with_base,
            "prefix_exact_match_with_base": self.prefix_exact_match_with_base,
            "new_divergence_rate_from_previous": self.new_divergence_rate_from_previous,
        }


def _orthogonal_centered_init(rows: int, n_experts: int, noise_init_std: float) -> torch.Tensor:
    """Sample seed rows that are zero-sum and pairwise orthogonal.

    Centered Gaussian rows are Gram-Schmidt orthogonalized inside the zero-sum
    expert-logit subspace, then rescaled so each row keeps the per-element std
    of the plain Gaussian init. If the subspace runs out of directions (only
    possible for tiny toy expert counts), the degenerate row falls back to a
    fresh centered Gaussian draw.
    """

    init = torch.randn(rows, n_experts)
    init = init - init.mean(dim=-1, keepdim=True)
    for row in range(rows):
        for prev in range(row):
            prev_vec = init[prev]
            denom = prev_vec.dot(prev_vec).clamp_min(1e-12)
            init[row] = init[row] - prev_vec * (init[row].dot(prev_vec) / denom)
        if float(init[row].norm()) < 1e-6:
            fallback = torch.randn(n_experts)
            init[row] = fallback - fallback.mean()
    target_norm = noise_init_std * math.sqrt(n_experts)
    return init * (target_norm / init.norm(dim=-1, keepdim=True).clamp_min(1e-12))


class SeedRouterNoise(nn.Module):
    """Trainable first-router trajectory seed.

    Row 0 is the unmodified base trajectory. Rows 1..N-1 receive trainable
    expert-logit bias only at the first MoE layer. Later layers are left to
    diverge naturally through their changed hidden states.
    """

    def __init__(
        self,
        num_trajectories: int,
        n_experts: int,
        noise_init_std: float = 0.02,
        noise_scale: float = 0.1,
        noise_scale_max: float = 1.0,
        target_layer_idx: Optional[int] = None,
        train_router_mode: str = "st_topk",
        soft_temperature: float = 1.0,
        disable_noise: bool = False,
        hidden_size: Optional[int] = None,
        context_seed_gate: bool = False,
        context_scale_max_delta: float = 0.5,
        seed_init_mode: str = "orthogonal",
        inject_mode: str = "first",
    ) -> None:
        super().__init__()
        if num_trajectories not in (3, 5):
            raise ValueError("num_trajectories must be one of {3, 5}")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if not 0.0 < noise_scale < noise_scale_max:
            raise ValueError("noise_scale must be in (0, noise_scale_max)")
        if train_router_mode not in ("hard", "soft_all", "st_topk"):
            raise ValueError("train_router_mode must be one of {'hard', 'soft_all', 'st_topk'}")
        if seed_init_mode not in ("gaussian", "orthogonal"):
            raise ValueError("seed_init_mode must be one of {'gaussian', 'orthogonal'}")
        if inject_mode not in ("first", "all"):
            raise ValueError("inject_mode must be one of {'first', 'all'}")
        if soft_temperature <= 0:
            raise ValueError("soft_temperature must be positive")
        if context_seed_gate and hidden_size is None:
            raise ValueError("hidden_size is required when context_seed_gate=True")
        if not 0.0 <= context_scale_max_delta < 1.0:
            raise ValueError("context_scale_max_delta must be in [0, 1)")

        self.num_trajectories = int(num_trajectories)
        self.n_experts = int(n_experts)
        self.hidden_size = int(hidden_size) if hidden_size is not None else None
        self.noise_scale = float(noise_scale)
        self.noise_scale_max = float(noise_scale_max)
        self.target_layer_idx = target_layer_idx
        self.train_router_mode = train_router_mode
        self.soft_temperature = float(soft_temperature)
        self.disable_noise = bool(disable_noise)
        self.context_seed_gate = bool(context_seed_gate)
        self.context_scale_max_delta = float(context_scale_max_delta)
        self.seed_init_mode = seed_init_mode
        self.inject_mode = inject_mode
        # "all" mode: per-layer seeds are registered once the MoE layers are
        # discovered during patching (register_inject_layers).
        self.inject_layer_ids: Optional[List[int]] = None
        self.layer_noise: Optional[nn.Parameter] = None
        self.raw_layer_noise_scale: Optional[nn.Parameter] = None
        self._noise_init_std = float(noise_init_std)
        self.noise = nn.Parameter(torch.empty(num_trajectories - 1, n_experts))
        raw_scale_init = math.log(noise_scale / (noise_scale_max - noise_scale))
        self.raw_noise_scale = nn.Parameter(
            torch.full((num_trajectories - 1, 1), float(raw_scale_init), dtype=torch.float32)
        )
        if self.context_seed_gate:
            self.context_norm = nn.LayerNorm(self.hidden_size)
            self.context_scale_proj = nn.Linear(self.hidden_size, num_trajectories - 1, bias=True)
        else:
            self.context_norm = None
            self.context_scale_proj = None
        self.record_routing = False
        self.record_token_routing = False
        self.record_mask: Optional[torch.Tensor] = None
        self.current_mask: Optional[torch.Tensor] = None
        # KV-cached decoding: the context gate must keep the prompt-level seed
        # multiplier instead of recomputing it from a single new token.
        self.frozen_context_multiplier: Optional[torch.Tensor] = None
        self.last_context_multiplier: Optional[torch.Tensor] = None
        self.layer_stats: Dict[int, RoutingLayerStats] = {}
        self.path_stats: Dict[int, RoutingPathStats] = {}
        self.token_routing: Dict[int, Dict[str, torch.Tensor]] = {}
        self._path_prefix_match: Optional[torch.Tensor] = None
        self.reset_parameters(noise_init_std)

    def reset_parameters(self, noise_init_std: float = 0.02) -> None:
        if self.seed_init_mode == "orthogonal":
            with torch.no_grad():
                self.noise.copy_(
                    _orthogonal_centered_init(
                        rows=self.num_trajectories - 1,
                        n_experts=self.n_experts,
                        noise_init_std=float(noise_init_std),
                    )
                )
        else:
            nn.init.normal_(self.noise, mean=0.0, std=float(noise_init_std))
        if self.context_scale_proj is not None:
            nn.init.zeros_(self.context_scale_proj.weight)
            nn.init.zeros_(self.context_scale_proj.bias)

    def clear_stats(self) -> None:
        self.layer_stats = {}
        self.path_stats = {}
        self.token_routing = {}
        self._path_prefix_match = None
        self.record_mask = None
        self.current_mask = None

    def register_inject_layers(self, layer_ids: List[int]) -> None:
        """Create per-layer seeds once MoE layers are known (inject_mode='all')."""

        self.inject_layer_ids = sorted(int(layer) for layer in layer_ids)
        if self.inject_mode != "all" or self.layer_noise is not None:
            return
        num_layers = len(self.inject_layer_ids)
        rows = self.num_trajectories - 1
        init = torch.stack(
            [
                _orthogonal_centered_init(rows, self.n_experts, self._noise_init_std)
                if self.seed_init_mode == "orthogonal"
                else (lambda t: t - t.mean(dim=-1, keepdim=True))(
                    torch.randn(rows, self.n_experts) * self._noise_init_std
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_noise = nn.Parameter(init)
        raw_scale_init = math.log(self.noise_scale / (self.noise_scale_max - self.noise_scale))
        self.raw_layer_noise_scale = nn.Parameter(
            torch.full((num_layers, rows, 1), float(raw_scale_init), dtype=torch.float32)
        )

    def _layer_slot(self, layer_idx: Optional[int]) -> Optional[int]:
        if self.inject_mode != "all" or self.inject_layer_ids is None or layer_idx is None:
            return None
        try:
            return self.inject_layer_ids.index(int(layer_idx))
        except ValueError:
            return None

    def centered_noise(self) -> torch.Tensor:
        return self.noise - self.noise.mean(dim=-1, keepdim=True)

    def noise_scale_value(self) -> torch.Tensor:
        return self.noise_scale_max * torch.sigmoid(self.raw_noise_scale.float())

    def effective_noise(self) -> torch.Tensor:
        if self.disable_noise:
            return torch.zeros_like(self.centered_noise().float())
        return self.centered_noise().float() * self.noise_scale_value()

    def effective_noise_for_layer(self, layer_idx: Optional[int]) -> torch.Tensor:
        slot = self._layer_slot(layer_idx)
        if slot is None or self.layer_noise is None or self.raw_layer_noise_scale is None:
            return self.effective_noise()
        if self.disable_noise:
            return torch.zeros(
                self.num_trajectories - 1, self.n_experts, device=self.layer_noise.device
            )
        noise = self.layer_noise[slot]
        centered = noise - noise.mean(dim=-1, keepdim=True)
        scale = self.noise_scale_max * torch.sigmoid(self.raw_layer_noise_scale[slot].float())
        return centered.float() * scale

    def noise_scale_value_for_layer(self, layer_idx: Optional[int]) -> torch.Tensor:
        slot = self._layer_slot(layer_idx)
        if slot is None or self.raw_layer_noise_scale is None:
            return self.noise_scale_value()
        return self.noise_scale_max * torch.sigmoid(self.raw_layer_noise_scale[slot].float())

    def context_multiplier(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.context_seed_gate or self.context_scale_proj is None or self.context_norm is None:
            return None
        batch_rows, seq_len, hidden_size = hidden_states.shape
        if batch_rows % self.num_trajectories != 0:
            return None
        batch = batch_rows // self.num_trajectories
        base_hidden = hidden_states.reshape(batch, self.num_trajectories, seq_len, hidden_size)[:, 0]
        base_hidden = base_hidden.float()
        if self.current_mask is not None:
            mask = self.current_mask.to(device=base_hidden.device, dtype=base_hidden.dtype)
            if tuple(mask.shape) != (batch, seq_len):
                raise ValueError(
                    f"current_mask must have shape {(batch, seq_len)}, got {tuple(mask.shape)}"
                )
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (base_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = base_hidden.mean(dim=1)
        gate_input = self.context_norm(pooled.to(self.context_scale_proj.weight.dtype))
        raw = self.context_scale_proj(gate_input).float()
        multiplier = 1.0 + self.context_scale_max_delta * torch.tanh(raw)
        return multiplier.clamp_min(1.0 - self.context_scale_max_delta)

    def full_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        zero = torch.zeros(1, self.n_experts, device=device, dtype=dtype)
        noisy = self.effective_noise().to(device=device, dtype=dtype)
        return torch.cat([zero, noisy], dim=0)

    def bias_for_hidden_states(
        self,
        hidden_states: torch.Tensor,
        dtype: torch.dtype,
        layer_idx: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[List[float]]]:
        batch_rows = hidden_states.shape[0]
        if batch_rows % self.num_trajectories != 0:
            return None, None
        batch = batch_rows // self.num_trajectories
        effective = self.effective_noise_for_layer(layer_idx).to(
            device=hidden_states.device, dtype=dtype
        )
        if self.frozen_context_multiplier is not None:
            multiplier = self.frozen_context_multiplier.to(device=hidden_states.device)
            if multiplier.shape[0] != batch:
                raise ValueError(
                    f"frozen_context_multiplier batch {multiplier.shape[0]} != {batch}"
                )
        else:
            multiplier = self.context_multiplier(hidden_states)
            self.last_context_multiplier = multiplier.detach() if multiplier is not None else None
        if multiplier is not None:
            effective_by_batch = effective.unsqueeze(0) * multiplier.to(dtype=dtype).unsqueeze(-1)
            scale_values = (
                self.noise_scale_value_for_layer(layer_idx).to(device=hidden_states.device).view(1, -1)
                * multiplier.detach().float()
            ).mean(dim=0)
        else:
            effective_by_batch = effective.unsqueeze(0).expand(batch, -1, -1)
            scale_values = (
                self.noise_scale_value_for_layer(layer_idx).to(device=hidden_states.device).view(-1)
            )

        zero = torch.zeros(batch, 1, self.n_experts, device=hidden_states.device, dtype=dtype)
        full = torch.cat([zero, effective_by_batch], dim=1)
        scale_with_base = torch.cat(
            [
                torch.zeros(1, device=scale_values.device, dtype=scale_values.dtype),
                scale_values.detach().float(),
            ],
            dim=0,
        )
        return full.reshape(batch_rows, self.n_experts), scale_with_base.cpu().tolist()

    @staticmethod
    def _cosine_collapse(effective: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(effective, dim=-1, eps=1e-6)
        cosine = normalized @ normalized.transpose(0, 1)
        eye = torch.eye(cosine.shape[0], device=cosine.device, dtype=torch.bool)
        off_diag = cosine.masked_select(~eye)
        if off_diag.numel() == 0:
            return cosine.sum() * 0.0
        return off_diag.pow(2).mean()

    def diversity_loss(self) -> torch.Tensor:
        """Penalize cosine collapse among non-base seed vectors."""

        if self.inject_mode == "all" and self.layer_noise is not None:
            losses = [
                self._cosine_collapse(self.effective_noise_for_layer(layer_idx))
                for layer_idx in (self.inject_layer_ids or [])
            ]
            if losses:
                return torch.stack(losses).mean()
        return self._cosine_collapse(self.effective_noise())

    def l2_loss(self) -> torch.Tensor:
        if self.inject_mode == "all" and self.layer_noise is not None:
            losses = [
                self.effective_noise_for_layer(layer_idx).pow(2).mean()
                for layer_idx in (self.inject_layer_ids or [])
            ]
            if losses:
                return torch.stack(losses).mean()
        return self.effective_noise().pow(2).mean()

    def context_gate_l2_loss(self) -> torch.Tensor:
        if self.context_scale_proj is None:
            return self.noise.sum() * 0.0
        return (
            self.context_scale_proj.weight.float().pow(2).mean()
            + self.context_scale_proj.bias.float().pow(2).mean()
        )

    def context_gate_norm(self) -> torch.Tensor:
        if self.context_scale_proj is None:
            return self.noise.sum().detach().float() * 0.0
        weight_norm = self.context_scale_proj.weight.detach().float().norm()
        bias_norm = self.context_scale_proj.bias.detach().float().norm()
        return torch.sqrt(weight_norm.pow(2) + bias_norm.pow(2))

    def bias_for_batch_rows(
        self,
        batch_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if batch_rows % self.num_trajectories != 0:
            return None
        traj_ids = torch.arange(batch_rows, device=device) % self.num_trajectories
        return self.full_bias(device, dtype).index_select(0, traj_ids)

    def should_inject(self, layer_idx: int) -> bool:
        if self.disable_noise:
            return False
        if self.inject_mode == "all":
            return self.inject_layer_ids is not None and int(layer_idx) in self.inject_layer_ids
        return self.target_layer_idx is not None and layer_idx == self.target_layer_idx

    def use_relaxed_training_router(self, layer_idx: int) -> bool:
        return (
            not self.disable_noise
            and self.training
            and self.train_router_mode in ("soft_all", "st_topk")
            and self.target_layer_idx is not None
            and layer_idx == self.target_layer_idx
        )

    def stats_as_dict(self) -> Dict[int, Dict[str, object]]:
        return {
            layer_idx: self.layer_stats[layer_idx].as_dict()
            for layer_idx in sorted(self.layer_stats)
        }

    def path_stats_as_dict(self) -> Dict[int, Dict[str, object]]:
        return {
            layer_idx: self.path_stats[layer_idx].as_dict()
            for layer_idx in sorted(self.path_stats)
        }


def _entropy(scores: torch.Tensor) -> torch.Tensor:
    return -(scores * scores.clamp_min(1e-20).log()).sum(dim=-1)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-20)
    q = q.clamp_min(1e-20)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    weighted = values * mask.to(dtype=values.dtype)
    numerator = weighted.sum(dim=dims)
    denominator = mask.to(dtype=values.dtype).sum(dim=dims).clamp_min(1.0)
    return numerator / denominator


def _record_valid_mask(
    record_mask: Optional[torch.Tensor],
    batch: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if record_mask is None:
        return torch.ones(batch, seq_len, device=device, dtype=torch.bool)
    mask = record_mask.to(device=device, dtype=torch.bool)
    if tuple(mask.shape) != (batch, seq_len):
        raise ValueError(
            f"record_mask must have shape {(batch, seq_len)}, got {tuple(mask.shape)}"
        )
    return mask


def _summarize_routing(
    logits: torch.Tensor,
    scores: torch.Tensor,
    topk_idx: torch.Tensor,
    layer_idx: int,
    noise_applied: bool,
    num_trajectories: int,
    n_experts: int,
    batch_rows: int,
    seq_len: int,
    valid_mask: torch.Tensor,
    seed_scale_by_traj: Optional[List[float]] = None,
) -> RoutingLayerStats:
    batch = batch_rows // num_trajectories
    k = topk_idx.shape[-1]

    logits_by_traj = logits.detach().float().reshape(batch, num_trajectories, seq_len, n_experts)
    scores_by_traj = scores.detach().float().reshape(batch, num_trajectories, seq_len, n_experts)
    topk_by_traj = topk_idx.detach().reshape(batch, num_trajectories, seq_len, k)
    valid_by_traj = valid_mask[:, None, :, None]
    valid_by_alt = valid_mask[:, None, :]

    entropy_by_traj = _masked_mean(
        _entropy(scores_by_traj),
        valid_by_traj.squeeze(-1),
        dims=(0, 2),
    ).cpu().tolist()
    expert_probs_by_traj = _masked_mean(
        scores_by_traj,
        valid_by_traj,
        dims=(0, 2),
    )
    expert_probs_by_traj_list = expert_probs_by_traj.cpu().tolist()
    expert_jsd_with_base = _js_divergence(
        expert_probs_by_traj[1:],
        expert_probs_by_traj[0:1].expand_as(expert_probs_by_traj[1:]),
    ).cpu().tolist()
    if n_experts > k:
        score_boundary = torch.topk(scores_by_traj, k=k + 1, dim=-1, sorted=True).values
        score_margin = score_boundary[..., k - 1] - score_boundary[..., k]
        topk_margin_by_traj = _masked_mean(
            score_margin,
            valid_by_traj.squeeze(-1),
            dims=(0, 2),
        ).cpu().tolist()
        logit_boundary = torch.topk(logits_by_traj, k=k + 1, dim=-1, sorted=True).values
        logit_margin = logit_boundary[..., k - 1] - logit_boundary[..., k]
        topk_logit_margin_by_traj = _masked_mean(
            logit_margin,
            valid_by_traj.squeeze(-1),
            dims=(0, 2),
        ).cpu().tolist()
    else:
        topk_margin_by_traj = [0.0 for _ in range(num_trajectories)]
        topk_logit_margin_by_traj = [0.0 for _ in range(num_trajectories)]

    base_one_hot = F.one_hot(topk_by_traj[:, 0], num_classes=n_experts).sum(dim=-2).clamp(max=1).bool()
    alt_one_hot = F.one_hot(topk_by_traj[:, 1:], num_classes=n_experts).sum(dim=-2).clamp(max=1).bool()
    overlap = (base_one_hot.unsqueeze(1) & alt_one_hot).float().sum(dim=-1) / float(k)
    topk_overlap = _masked_mean(overlap, valid_by_alt, dims=(0, 2)).cpu().tolist()
    exact_match = (base_one_hot.unsqueeze(1) == alt_one_hot).all(dim=-1).float()
    topk_exact_match = _masked_mean(exact_match, valid_by_alt, dims=(0, 2)).cpu().tolist()

    return RoutingLayerStats(
        layer_idx=layer_idx,
        noise_applied=noise_applied,
        entropy_by_traj=entropy_by_traj,
        topk_overlap_with_base=topk_overlap,
        topk_exact_match_with_base=topk_exact_match,
        topk_margin_by_traj=topk_margin_by_traj,
        topk_logit_margin_by_traj=topk_logit_margin_by_traj,
        expert_probs_by_traj=expert_probs_by_traj_list,
        expert_jsd_with_base=expert_jsd_with_base,
        seed_scale_by_traj=seed_scale_by_traj or [],
    )


def _update_path_stats(
    router_noise: SeedRouterNoise,
    hard_topk_idx: torch.Tensor,
    layer_idx: int,
    batch_rows: int,
    seq_len: int,
    valid_mask: torch.Tensor,
) -> None:
    batch = batch_rows // router_noise.num_trajectories
    k = hard_topk_idx.shape[-1]
    topk_by_traj = hard_topk_idx.detach().reshape(
        batch,
        router_noise.num_trajectories,
        seq_len,
        k,
    )
    topk_sets = torch.sort(topk_by_traj, dim=-1).values
    layer_match = (topk_sets[:, 0:1] == topk_sets[:, 1:]).all(dim=-1)

    if (
        router_noise._path_prefix_match is None
        or tuple(router_noise._path_prefix_match.shape) != tuple(layer_match.shape)
    ):
        previous_prefix = torch.ones_like(layer_match, dtype=torch.bool)
    else:
        previous_prefix = router_noise._path_prefix_match.to(device=layer_match.device)

    new_divergence = previous_prefix & ~layer_match
    prefix_match = previous_prefix & layer_match
    router_noise._path_prefix_match = prefix_match

    valid_by_alt = valid_mask[:, None, :].expand_as(layer_match)
    router_noise.path_stats[layer_idx] = RoutingPathStats(
        layer_idx=layer_idx,
        layer_exact_match_with_base=_masked_mean(
            layer_match.float(),
            valid_by_alt,
            dims=(0, 2),
        ).cpu().tolist(),
        prefix_exact_match_with_base=_masked_mean(
            prefix_match.float(),
            valid_by_alt,
            dims=(0, 2),
        ).cpu().tolist(),
        new_divergence_rate_from_previous=_masked_mean(
            new_divergence.float(),
            valid_by_alt,
            dims=(0, 2),
        ).cpu().tolist(),
    )


def _trajectory_gate_forward(self, hidden_states):
    """MoEGate.forward replacement with optional first-layer trajectory bias."""

    batch_rows, seq_len, hidden = hidden_states.shape
    flat_hidden = hidden_states.view(-1, hidden)
    logits = F.linear(flat_hidden, self.weight, None)

    router_noise: Optional[SeedRouterNoise] = getattr(self, "_trajectory_router_noise", None)
    layer_idx: Optional[int] = getattr(self, "_trajectory_layer_idx", None)
    noise_applied = False
    seed_scale_by_traj: Optional[List[float]] = None

    if router_noise is not None and layer_idx is not None and router_noise.should_inject(layer_idx):
        row_bias, seed_scale_by_traj = router_noise.bias_for_hidden_states(
            hidden_states, logits.dtype, layer_idx=layer_idx
        )
        if row_bias is not None:
            logits = logits + row_bias.repeat_interleave(seq_len, dim=0)
            noise_applied = True

    if self.scoring_func == "softmax":
        scores = logits.softmax(dim=-1)
    else:
        raise NotImplementedError(f"unsupported MoE scoring function: {self.scoring_func}")

    hard_topk_weight, hard_topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
    if self.top_k > 1 and self.norm_topk_prob:
        hard_topk_weight = hard_topk_weight / (hard_topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

    if router_noise is not None and layer_idx is not None and router_noise.use_relaxed_training_router(layer_idx):
        expert_count = scores.shape[-1]
        topk_idx = torch.arange(expert_count, device=scores.device).expand(scores.shape[0], expert_count)
        soft_weight = torch.softmax(logits.float() / router_noise.soft_temperature, dim=-1).to(scores.dtype)
        hard_full = torch.zeros_like(soft_weight)
        hard_full.scatter_add_(dim=-1, index=hard_topk_idx, src=hard_topk_weight)
        if not self.norm_topk_prob:
            target_mass = hard_topk_weight.detach().sum(dim=-1, keepdim=True)
            soft_weight = soft_weight * target_mass
        if router_noise.train_router_mode == "soft_all":
            topk_weight = soft_weight
        else:
            topk_weight = hard_full.detach() + soft_weight - soft_weight.detach()
    else:
        topk_weight, topk_idx = hard_topk_weight, hard_topk_idx

    if (
        router_noise is not None
        and router_noise.record_token_routing
        and layer_idx is not None
        and batch_rows % router_noise.num_trajectories == 0
    ):
        batch = batch_rows // router_noise.num_trajectories
        k = hard_topk_idx.shape[-1]
        router_noise.token_routing[layer_idx] = {
            "topk_idx": hard_topk_idx.detach()
            .reshape(batch, router_noise.num_trajectories, seq_len, k)
            .cpu(),
            "topk_weight": hard_topk_weight.detach()
            .float()
            .reshape(batch, router_noise.num_trajectories, seq_len, k)
            .cpu(),
        }

    if (
        router_noise is not None
        and router_noise.record_routing
        and layer_idx is not None
        and batch_rows % router_noise.num_trajectories == 0
    ):
        valid_mask = _record_valid_mask(
            router_noise.record_mask,
            batch=batch_rows // router_noise.num_trajectories,
            seq_len=seq_len,
            device=scores.device,
        )
        router_noise.layer_stats[layer_idx] = _summarize_routing(
            logits=logits,
            scores=scores,
            topk_idx=hard_topk_idx,
            layer_idx=layer_idx,
            noise_applied=noise_applied,
            num_trajectories=router_noise.num_trajectories,
            n_experts=router_noise.n_experts,
            batch_rows=batch_rows,
            seq_len=seq_len,
            valid_mask=valid_mask,
            seed_scale_by_traj=seed_scale_by_traj if noise_applied else None,
        )
        _update_path_stats(
            router_noise=router_noise,
            hard_topk_idx=hard_topk_idx,
            layer_idx=layer_idx,
            batch_rows=batch_rows,
            seq_len=seq_len,
            valid_mask=valid_mask,
        )

    # The wrapped backbone is kept in eval mode, so DeepSeek's aux loss is not
    # used by this adapter path.
    return topk_idx, topk_weight, None


def _differentiable_moe_infer(
    moe_module,
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
) -> torch.Tensor:
    """MoE inference path that preserves gradients to selected gate weights."""

    tokens, hidden = hidden_states.shape
    k = topk_idx.shape[-1]
    flat_expert_idx = topk_idx.reshape(-1)
    flat_weight = topk_weight.reshape(-1, 1)
    token_idx = torch.arange(tokens, device=hidden_states.device).repeat_interleave(k)
    output = torch.zeros_like(hidden_states)

    order = torch.argsort(flat_expert_idx)
    sorted_expert = flat_expert_idx.index_select(0, order)
    sorted_token = token_idx.index_select(0, order)
    sorted_weight = flat_weight.index_select(0, order)
    unique_expert, counts = torch.unique_consecutive(sorted_expert, return_counts=True)
    boundaries = torch.cumsum(counts, dim=0)

    start = 0
    for expert_id, end in zip(unique_expert.tolist(), boundaries.tolist()):
        selected_tokens = sorted_token[start:end]
        expert_input = hidden_states.index_select(0, selected_tokens)
        expert_output = moe_module.experts[expert_id](expert_input).to(output.dtype)
        expert_output = expert_output * sorted_weight[start:end]
        output.index_add_(0, selected_tokens, expert_output)
        start = end
    return output


def _trajectory_moe_forward(self, hidden_states):
    """DeepseekMoE.forward replacement without the original no_grad infer path."""

    identity = hidden_states
    original_shape = hidden_states.shape
    topk_idx, topk_weight, _ = self.gate(hidden_states)
    flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
    routed = _differentiable_moe_infer(self, flat_hidden, topk_idx, topk_weight).view(*original_shape)
    if self.config.n_shared_experts is not None:
        routed = routed + self.shared_experts(identity)
    return routed


def _layer_index_from_name(name: str) -> Optional[int]:
    marker = ".layers."
    if marker not in name:
        return None
    try:
        return int(name.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def patch_deepseek_moe_gates(model: nn.Module, router_noise: SeedRouterNoise) -> List[int]:
    """Patch DeepSeek MoE gates/blocks and return the MoE layer indices found."""

    patched_layers: List[int] = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in ("MoEGate", "DeepseekMoE"):
            continue
        layer_idx = _layer_index_from_name(name)
        if layer_idx is None:
            continue
        if module_type == "MoEGate":
            n_experts = int(getattr(module, "n_routed_experts", router_noise.n_experts))
            if n_experts != router_noise.n_experts:
                raise ValueError(
                    f"gate {name} has {n_experts} experts, expected {router_noise.n_experts}"
                )
            module._trajectory_router_noise = router_noise
            module._trajectory_layer_idx = layer_idx
            if not hasattr(module, "_trajectory_original_forward"):
                module._trajectory_original_forward = module.forward
            module.forward = types.MethodType(_trajectory_gate_forward, module)
            patched_layers.append(layer_idx)
        else:
            if not hasattr(module, "_trajectory_original_forward"):
                module._trajectory_original_forward = module.forward
            module.forward = types.MethodType(_trajectory_moe_forward, module)

    if not patched_layers:
        raise RuntimeError("no DeepSeek MoEGate modules were found to patch")

    first_layer = min(patched_layers)
    if router_noise.target_layer_idx is None:
        router_noise.target_layer_idx = first_layer
    elif router_noise.target_layer_idx not in patched_layers:
        raise ValueError(
            f"target_layer_idx={router_noise.target_layer_idx} was not found in patched MoE layers"
        )
    router_noise.register_inject_layers(sorted(patched_layers))
    return sorted(patched_layers)
