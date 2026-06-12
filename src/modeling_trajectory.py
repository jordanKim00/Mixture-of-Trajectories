from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .aggregator import BaseAnchoredResidualAggregator
from .router_noise import SeedRouterNoise, patch_deepseek_moe_gates


@dataclass
class TrajectoryCausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    base_logits: Optional[torch.Tensor] = None
    route_stats: Optional[Dict[int, Dict[str, object]]] = None
    path_stats: Optional[Dict[int, Dict[str, object]]] = None
    aggregator_stats: Optional[Dict[str, object]] = None
    trajectory_stats: Optional[Dict[str, object]] = None
    trajectory_prediction_stats: Optional[Dict[str, object]] = None
    loss_components: Optional[Dict[str, object]] = None
    token_routing: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    pre_norm_by_traj: Optional[torch.Tensor] = None
    aggregator_alpha: Optional[torch.Tensor] = None


class TrajectoryEnsembleForCausalLM(nn.Module):
    """DeepSeek-MoE wrapper for seeded latent trajectory aggregation.

    The wrapped DeepSeek model stays frozen. Inputs are expanded from ``B`` to
    ``B * N`` rows, the first MoE router receives trajectory-specific seed
    noise, and final pre-norm hidden states are fused before the original final
    norm and LM head.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_trajectories: int = 3,
        agg_dim: int = 256,
        noise_init_std: float = 0.02,
        noise_scale: float = 0.1,
        noise_scale_max: float = 1.0,
        top_k: int = 6,
        train_router_mode: str = "st_topk",
        soft_temperature: float = 1.0,
        residual_scale_max: float = 0.25,
        include_null_aggregation_candidate: bool = True,
        aggregator_value_mode: str = "delta",
        disable_seed_noise: bool = False,
        context_seed_gate: bool = True,
        context_scale_max_delta: float = 0.5,
        seed_init_mode: str = "orthogonal",
        aggregator_relative_keys: bool = False,
    ) -> None:
        super().__init__()
        if num_trajectories not in (3, 5):
            raise ValueError("num_trajectories must be one of {3, 5}")

        self.base_model = base_model
        self.deepseek_model = base_model.model
        self.lm_head = base_model.lm_head
        self.config = base_model.config
        self.num_trajectories = int(num_trajectories)
        self.top_k = int(top_k)

        hidden_size = int(base_model.config.hidden_size)
        n_experts = int(base_model.config.n_routed_experts)
        self.router_noise = SeedRouterNoise(
            num_trajectories=num_trajectories,
            n_experts=n_experts,
            noise_init_std=noise_init_std,
            noise_scale=noise_scale,
            noise_scale_max=noise_scale_max,
            train_router_mode=train_router_mode,
            soft_temperature=soft_temperature,
            disable_noise=disable_seed_noise,
            hidden_size=hidden_size,
            context_seed_gate=context_seed_gate,
            context_scale_max_delta=context_scale_max_delta,
            seed_init_mode=seed_init_mode,
        )
        self.aggregator = BaseAnchoredResidualAggregator(
            hidden_size=hidden_size,
            agg_dim=agg_dim,
            residual_scale_init=0.01,
            residual_scale_max=residual_scale_max,
            include_null_candidate=include_null_aggregation_candidate,
            value_mode=aggregator_value_mode,
            relative_keys=aggregator_relative_keys,
        )

        self._freeze_backbone()
        self._set_top_k(top_k)
        self.patched_moe_layers = patch_deepseek_moe_gates(base_model, self.router_noise)
        self._place_trainables()
        self.base_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "deepseek-ai/deepseek-moe-16b-chat",
        num_trajectories: int = 3,
        agg_dim: int = 256,
        noise_init_std: float = 0.02,
        noise_scale: float = 0.1,
        noise_scale_max: float = 1.0,
        top_k: int = 6,
        train_router_mode: str = "st_topk",
        soft_temperature: float = 1.0,
        residual_scale_max: float = 0.25,
        include_null_aggregation_candidate: bool = True,
        aggregator_value_mode: str = "delta",
        disable_seed_noise: bool = False,
        context_seed_gate: bool = True,
        context_scale_max_delta: float = 0.5,
        seed_init_mode: str = "orthogonal",
        aggregator_relative_keys: bool = False,
        local_files_only: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> "TrajectoryEnsembleForCausalLM":
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return cls(
            base_model=base_model,
            num_trajectories=num_trajectories,
            agg_dim=agg_dim,
            noise_init_std=noise_init_std,
            noise_scale=noise_scale,
            noise_scale_max=noise_scale_max,
            top_k=top_k,
            train_router_mode=train_router_mode,
            soft_temperature=soft_temperature,
            residual_scale_max=residual_scale_max,
            include_null_aggregation_candidate=include_null_aggregation_candidate,
            aggregator_value_mode=aggregator_value_mode,
            disable_seed_noise=disable_seed_noise,
            context_seed_gate=context_seed_gate,
            context_scale_max_delta=context_scale_max_delta,
            seed_init_mode=seed_init_mode,
            aggregator_relative_keys=aggregator_relative_keys,
        )

    @property
    def input_device(self) -> torch.device:
        return self.deepseek_model.embed_tokens.weight.device

    @property
    def output_device(self) -> torch.device:
        return self.lm_head.weight.device

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.aggregator.train(mode)
        self.router_noise.train(mode)
        return self

    def _freeze_backbone(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad_(False)

    def _set_top_k(self, top_k: int) -> None:
        for module in self.base_model.modules():
            if type(module).__name__ == "MoEGate":
                module.top_k = int(top_k)
            elif type(module).__name__ == "DeepseekMoE":
                module.num_experts_per_tok = int(top_k)

    def _place_trainables(self) -> None:
        first_gate_device = None
        for name, module in self.base_model.named_modules():
            if type(module).__name__ == "MoEGate" and f".layers.{self.router_noise.target_layer_idx}." in name:
                first_gate_device = module.weight.device
                break
        if first_gate_device is None:
            first_gate_device = self.input_device
        self.router_noise.to(device=first_gate_device)

        final_norm_weight = next(self.deepseek_model.norm.parameters())
        self.aggregator.to(device=final_norm_weight.device)

    def trainable_parameters(self):
        if not self.router_noise.disable_noise:
            yield from self.router_noise.parameters()
        yield from self.aggregator.parameters()

    def adapter_config(self) -> Dict[str, object]:
        return {
            "num_trajectories": self.num_trajectories,
            "agg_dim": self.aggregator.agg_dim,
            "hidden_size": self.aggregator.hidden_size,
            "noise_scale": self.router_noise.noise_scale,
            "noise_scale_max": self.router_noise.noise_scale_max,
            "noise_scale_value": self.router_noise.noise_scale_value().detach().cpu().view(-1).tolist(),
            "disable_seed_noise": self.router_noise.disable_noise,
            "context_seed_gate": self.router_noise.context_seed_gate,
            "context_scale_max_delta": self.router_noise.context_scale_max_delta,
            "train_router_mode": self.router_noise.train_router_mode,
            "soft_temperature": self.router_noise.soft_temperature,
            "target_layer_idx": self.router_noise.target_layer_idx,
            "top_k": self.top_k,
            "patched_moe_layers": self.patched_moe_layers,
            "residual_scale_max": self.aggregator.residual_scale_max,
            "include_null_aggregation_candidate": self.aggregator.include_null_candidate,
            "aggregator_value_mode": self.aggregator.value_mode,
            "aggregator_relative_keys": self.aggregator.relative_keys,
            "seed_init_mode": self.router_noise.seed_init_mode,
        }

    def adapter_regularization(
        self,
        noise_diversity_weight: float = 0.0,
        noise_l2_weight: float = 0.0,
        context_gate_l2_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        device = self.router_noise.noise.device
        total = torch.zeros((), device=device, dtype=torch.float32)
        metrics: Dict[str, object] = {}
        if self.router_noise.disable_noise:
            metrics["seed_noise_disabled"] = True
            metrics["noise_scale_value_mean"] = 0.0
            metrics["noise_scale_value_min"] = 0.0
            metrics["noise_scale_value_max"] = 0.0
            metrics["context_gate_l2"] = 0.0
            metrics["context_gate_norm"] = 0.0
            metrics["adapter_reg_total"] = 0.0
            return total, metrics
        if noise_diversity_weight > 0:
            diversity = self.router_noise.diversity_loss()
            total = total + float(noise_diversity_weight) * diversity
            metrics["noise_diversity"] = float(diversity.detach().cpu().item())
            metrics["noise_diversity_weight"] = float(noise_diversity_weight)
        if noise_l2_weight > 0:
            l2 = self.router_noise.l2_loss()
            total = total + float(noise_l2_weight) * l2
            metrics["noise_l2"] = float(l2.detach().cpu().item())
            metrics["noise_l2_weight"] = float(noise_l2_weight)
        context_gate_l2 = self.router_noise.context_gate_l2_loss()
        if context_gate_l2_weight > 0:
            total = total + float(context_gate_l2_weight) * context_gate_l2
            metrics["context_gate_l2_weight"] = float(context_gate_l2_weight)
        metrics["context_gate_l2"] = float(context_gate_l2.detach().cpu().item())
        metrics["context_gate_norm"] = float(self.router_noise.context_gate_norm().cpu().item())
        scale = self.router_noise.noise_scale_value().detach().float().cpu().view(-1)
        metrics["noise_scale_value_mean"] = float(scale.mean().item())
        metrics["noise_scale_value_min"] = float(scale.min().item())
        metrics["noise_scale_value_max"] = float(scale.max().item())
        metrics["adapter_reg_total"] = float(total.detach().cpu().item())
        return total, metrics

    def save_adapter(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "config": self.adapter_config(),
                "router_noise": self.router_noise.state_dict(),
                "aggregator": self.aggregator.state_dict(),
            },
            path,
        )

    def load_adapter(self, path: str, map_location: Optional[str] = None) -> Dict[str, object]:
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint.get("config", {})
        checkpoint_uses_context_gate = bool(
            config.get("context_seed_gate", self.router_noise.context_seed_gate)
        )
        if checkpoint_uses_context_gate and self.router_noise.context_scale_proj is None:
            raise ValueError(
                "checkpoint uses context_seed_gate=True; construct the wrapper with "
                "context_seed_gate=True before loading"
            )
        checkpoint_uses_null_candidate = bool(
            config.get(
                "include_null_aggregation_candidate",
                self.aggregator.include_null_candidate,
            )
        )
        if checkpoint_uses_null_candidate and self.aggregator.null_key is None:
            raise ValueError(
                "checkpoint uses include_null_aggregation_candidate=True; construct "
                "the wrapper with include_null_aggregation_candidate=True before loading"
            )
        self.router_noise.load_state_dict(checkpoint["router_noise"], strict=False)
        self.aggregator.load_state_dict(checkpoint["aggregator"], strict=False)
        if "include_null_aggregation_candidate" in config:
            self.aggregator.include_null_candidate = bool(
                config["include_null_aggregation_candidate"]
            )
        if "aggregator_value_mode" in config:
            self.aggregator.value_mode = str(config["aggregator_value_mode"])
        if "aggregator_relative_keys" in config:
            self.aggregator.relative_keys = bool(config["aggregator_relative_keys"])
        if "disable_seed_noise" in config:
            self.router_noise.disable_noise = bool(config["disable_seed_noise"])
        if "context_seed_gate" in config:
            self.router_noise.context_seed_gate = bool(config["context_seed_gate"])
        if "context_scale_max_delta" in config:
            self.router_noise.context_scale_max_delta = float(config["context_scale_max_delta"])
        return config

    def _expand_by_trajectory(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tensor.shape[:2]
        expanded = tensor[:, None, ...].expand(batch, self.num_trajectories, *tensor.shape[1:])
        return expanded.reshape(batch * self.num_trajectories, *tensor.shape[1:])

    def _prepare_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        seq_len = input_ids.shape[1]
        return torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        inputs_embeds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.deepseek_model._use_flash_attention_2:
            has_padding = attention_mask is not None and bool((attention_mask == 0).any().item())
            return attention_mask if has_padding else None

        module = sys.modules[self.deepseek_model.__class__.__module__]
        if self.deepseek_model._use_sdpa:
            return module._prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_len),
                inputs_embeds,
                0,
            )
        return module._prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_len),
            inputs_embeds,
            0,
        )

    def _run_deepseek_pre_norm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.input_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.input_device)
        if position_ids is not None:
            position_ids = position_ids.to(self.input_device)

        inputs_embeds = self.deepseek_model.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape
        prepared_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            inputs_embeds=inputs_embeds,
        )
        position_ids = self._prepare_position_ids(input_ids, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.deepseek_model.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=prepared_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        base_logits: torch.Tensor,
        kl_beta: float,
        kl_direction: str,
        kl_token_weight: Optional[torch.Tensor] = None,
        residual_l2: Optional[torch.Tensor] = None,
        residual_l2_weight: float = 0.0,
        trajectory_oracle_aux_loss: Optional[torch.Tensor] = None,
        trajectory_oracle_aux_weight: float = 0.0,
        trajectory_oracle_aux_components: Optional[Dict[str, object]] = None,
        aggregator_oracle_align_loss: Optional[torch.Tensor] = None,
        aggregator_oracle_align_weight: float = 0.0,
        aggregator_oracle_align_components: Optional[Dict[str, object]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if kl_direction not in ("base_to_fused", "symmetric"):
            raise ValueError("kl_direction must be one of {'base_to_fused', 'symmetric'}")
        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_labels = shift_labels.reshape(-1)
        valid_labels = flat_labels.ne(-100)

        if valid_labels.any():
            flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
            ce_loss = F.cross_entropy(flat_logits[valid_labels], flat_labels[valid_labels])
        else:
            ce_loss = shift_logits.float().sum() * 0.0
        loss = ce_loss
        shift_base_for_metrics = base_logits[..., :-1, :].contiguous().detach().to(logits.device)
        if valid_labels.any():
            flat_base = shift_base_for_metrics.reshape(-1, shift_base_for_metrics.shape[-1])
            base_ce = F.cross_entropy(flat_base[valid_labels], flat_labels[valid_labels])
        else:
            base_ce = shift_base_for_metrics.float().sum() * 0.0
        components = {
            "ce": float(ce_loss.detach().float().cpu().item()),
            "base_ce": float(base_ce.detach().float().cpu().item()),
            "ce_delta_vs_base": float((ce_loss - base_ce).detach().float().cpu().item()),
        }

        if kl_beta > 0:
            fused_log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            base_probs = F.softmax(shift_base_for_metrics.float(), dim=-1)
            base_to_fused = F.kl_div(fused_log_probs, base_probs, reduction="none").sum(dim=-1)
            if kl_direction == "symmetric":
                fused_probs = F.softmax(shift_logits.float(), dim=-1)
                base_log_probs = F.log_softmax(shift_base_for_metrics.float(), dim=-1)
                fused_to_base = F.kl_div(base_log_probs, fused_probs, reduction="none").sum(dim=-1)
                kl_per_token = 0.5 * (base_to_fused + fused_to_base)
            else:
                kl_per_token = base_to_fused
            valid = shift_labels.ne(-100)
            if kl_token_weight is not None:
                weight = kl_token_weight.detach().to(
                    device=kl_per_token.device, dtype=kl_per_token.dtype
                )
                if tuple(weight.shape) != tuple(kl_per_token.shape):
                    raise ValueError(
                        f"kl_token_weight must have shape {tuple(kl_per_token.shape)}, "
                        f"got {tuple(weight.shape)}"
                    )
                if valid.any():
                    weight_valid = weight[valid]
                    kl_loss = (kl_per_token[valid] * weight_valid).sum() / weight_valid.sum().clamp_min(
                        1e-12
                    )
                    components["kl_token_weight_mean"] = float(
                        weight_valid.mean().detach().float().cpu().item()
                    )
                else:
                    kl_loss = kl_per_token.sum() * 0.0
                    components["kl_token_weight_mean"] = 0.0
            elif valid.any():
                kl_loss = kl_per_token[valid].mean()
            else:
                kl_loss = kl_per_token.sum() * 0.0
            loss = loss + float(kl_beta) * kl_loss
            components["kl"] = float(kl_loss.detach().float().cpu().item())
            components["kl_beta"] = float(kl_beta)
            components["kl_direction"] = kl_direction
        else:
            components["kl"] = 0.0
            components["kl_beta"] = 0.0
        if residual_l2 is not None and residual_l2_weight > 0:
            loss = loss + float(residual_l2_weight) * residual_l2.to(loss.device)
            components["residual_l2"] = float(residual_l2.detach().float().cpu().item())
            components["residual_l2_weight"] = float(residual_l2_weight)
        elif residual_l2 is not None:
            components["residual_l2"] = float(residual_l2.detach().float().cpu().item())
            components["residual_l2_weight"] = 0.0
        if trajectory_oracle_aux_loss is not None and trajectory_oracle_aux_weight > 0:
            loss = loss + float(trajectory_oracle_aux_weight) * trajectory_oracle_aux_loss.to(loss.device)
            components["trajectory_oracle_aux_weight"] = float(trajectory_oracle_aux_weight)
            if trajectory_oracle_aux_components:
                components.update(trajectory_oracle_aux_components)
        elif trajectory_oracle_aux_components:
            components.update(trajectory_oracle_aux_components)
            components["trajectory_oracle_aux_weight"] = 0.0
        if aggregator_oracle_align_loss is not None and aggregator_oracle_align_weight > 0:
            loss = loss + float(aggregator_oracle_align_weight) * aggregator_oracle_align_loss.to(loss.device)
            components["aggregator_oracle_align_weight"] = float(aggregator_oracle_align_weight)
            if aggregator_oracle_align_components:
                components.update(aggregator_oracle_align_components)
        elif aggregator_oracle_align_components:
            components.update(aggregator_oracle_align_components)
            components["aggregator_oracle_align_weight"] = 0.0
        components["total"] = float(loss.detach().float().cpu().item())
        return loss, components

    @staticmethod
    def _masked_residual_l2_ratio(
        fused_pre_norm: torch.Tensor,
        base_pre_norm: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = (fused_pre_norm - base_pre_norm).float()
        base = base_pre_norm.float()
        ratio = residual.pow(2).sum(dim=-1) / base.pow(2).sum(dim=-1).clamp_min(1e-12)
        if attention_mask is not None:
            valid = attention_mask.to(device=ratio.device, dtype=torch.bool)
        else:
            valid = torch.ones_like(ratio, dtype=torch.bool)
        if valid.any():
            return ratio[valid].mean()
        return ratio.sum() * 0.0

    def _trajectory_oracle_auxiliary_loss(
        self,
        pre_norm_by_traj: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.5,
        include_base: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Multiple-choice CE over trajectory logits.

        The fused adapter is still trained by the main CE. This auxiliary term
        gives the first-router seed direct task credit by asking the noisy
        trajectories to contain at least one good gold-token predictor.
        """

        if temperature <= 0:
            raise ValueError("trajectory_oracle_temperature must be positive")
        start_idx = 0 if include_base else 1
        if start_idx >= pre_norm_by_traj.shape[1]:
            zero = pre_norm_by_traj.float().sum() * 0.0
            return zero, {
                "trajectory_oracle_aux_ce": 0.0,
                "trajectory_oracle_temperature": float(temperature),
                "trajectory_oracle_include_base": bool(include_base),
                "trajectory_oracle_num_candidates": 0,
            }

        shift_labels_source = labels[:, 1:].contiguous()
        if shift_labels_source.numel() == 0:
            zero = pre_norm_by_traj.float().sum() * 0.0
            return zero, {
                "trajectory_oracle_aux_ce": 0.0,
                "trajectory_oracle_temperature": float(temperature),
                "trajectory_oracle_include_base": bool(include_base),
                "trajectory_oracle_num_candidates": int(pre_norm_by_traj.shape[1] - start_idx),
            }

        nll_by_traj = []
        ce_by_candidate = []
        for traj_idx in range(start_idx, pre_norm_by_traj.shape[1]):
            traj_hidden = self.deepseek_model.norm(pre_norm_by_traj[:, traj_idx])
            traj_logits = self.lm_head(traj_hidden).float()
            shift_logits = traj_logits[:, :-1, :].contiguous()
            shift_labels = shift_labels_source.to(shift_logits.device)
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.clamp_min(0)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            nll_by_traj.append(nll)
            if valid.any():
                ce_by_candidate.append(float(nll[valid].detach().mean().cpu().item()))
            else:
                ce_by_candidate.append(0.0)

        stacked_nll = torch.stack(nll_by_traj, dim=1)
        valid = shift_labels_source.to(stacked_nll.device).ne(-100)
        if not valid.any():
            aux = stacked_nll.sum() * 0.0
        elif stacked_nll.shape[1] == 1:
            aux = stacked_nll[:, 0][valid].mean()
        else:
            tau = float(temperature)
            soft_oracle = -tau * torch.logsumexp(-stacked_nll / tau, dim=1)
            soft_oracle = soft_oracle + tau * math.log(stacked_nll.shape[1])
            aux = soft_oracle[valid].mean()

        return aux, {
            "trajectory_oracle_aux_ce": float(aux.detach().float().cpu().item()),
            "trajectory_oracle_candidate_ce": ce_by_candidate,
            "trajectory_oracle_temperature": float(temperature),
            "trajectory_oracle_include_base": bool(include_base),
            "trajectory_oracle_num_candidates": int(stacked_nll.shape[1]),
        }

    def _trajectory_nll_and_top1(
        self,
        pre_norm_by_traj: torch.Tensor,
        labels: torch.Tensor,
        detach_to_cpu: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels.to(self.output_device)
        shift_labels = labels[:, 1:].contiguous()
        valid = shift_labels.ne(-100)
        safe_labels = shift_labels.clamp_min(0)

        nll_by_traj = []
        top1_by_traj = []
        for traj_idx in range(pre_norm_by_traj.shape[1]):
            traj_pre_norm = pre_norm_by_traj[:, traj_idx]
            traj_hidden = self.deepseek_model.norm(traj_pre_norm)
            traj_logits = self.lm_head(traj_hidden).float()
            shift_logits = traj_logits[:, :-1, :].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            top1 = shift_logits.argmax(dim=-1)
            if detach_to_cpu:
                nll = nll.detach().float().cpu()
                top1 = top1.detach().cpu()
            nll_by_traj.append(nll)
            top1_by_traj.append(top1)

        valid_out = valid.detach().cpu() if detach_to_cpu else valid
        return torch.stack(nll_by_traj, dim=1), torch.stack(top1_by_traj, dim=1), valid_out

    def _aggregator_oracle_alignment_loss(
        self,
        pre_norm_by_traj: torch.Tensor,
        labels: torch.Tensor,
        aggregator_alpha: torch.Tensor,
        temperature: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if temperature <= 0:
            raise ValueError("aggregator_oracle_align_temperature must be positive")
        with torch.no_grad():
            nll_by_traj, _, valid = self._trajectory_nll_and_top1(
                pre_norm_by_traj=pre_norm_by_traj,
                labels=labels,
                detach_to_cpu=False,
            )
        if nll_by_traj.shape[-1] == 0:
            zero = aggregator_alpha.float().sum() * 0.0
            return zero, {
                "aggregator_oracle_align_ce": 0.0,
                "aggregator_oracle_align_temperature": float(temperature),
            }

        alt_nll = nll_by_traj[:, 1:].float()
        candidates = alt_nll.shape[1]
        alpha = aggregator_alpha[:, :, : alt_nll.shape[-1]].float()
        if alpha.shape[1] == candidates + 1:
            base_nll = nll_by_traj[:, 0:1].float()
            target_scores = torch.cat([-alt_nll / float(temperature), -base_nll / float(temperature)], dim=1)
            has_null = True
        elif alpha.shape[1] == candidates:
            target_scores = -alt_nll / float(temperature)
            has_null = False
        else:
            raise ValueError(
                f"aggregator_alpha has {alpha.shape[1]} candidates, expected {candidates} or {candidates + 1}"
            )
        target_probs = torch.softmax(target_scores, dim=1).detach()
        per_token = -(target_probs * alpha.clamp_min(1e-20).log()).sum(dim=1)
        if valid.any():
            align_loss = per_token[valid].mean()
        else:
            align_loss = per_token.sum() * 0.0
        return align_loss, {
            "aggregator_oracle_align_ce": float(align_loss.detach().float().cpu().item()),
            "aggregator_oracle_align_temperature": float(temperature),
            "aggregator_oracle_align_has_null": has_null,
        }

    def _trajectory_stats(
        self,
        pre_norm_by_traj: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        base = pre_norm_by_traj[:, 0].detach().float()
        alt = pre_norm_by_traj[:, 1:].detach().float()
        if attention_mask is not None:
            mask = attention_mask.to(device=base.device, dtype=torch.bool)
        else:
            mask = torch.ones(base.shape[:2], device=base.device, dtype=torch.bool)

        base_expanded = base.unsqueeze(1).expand_as(alt)
        cosine = F.cosine_similarity(base_expanded, alt, dim=-1)
        delta_ratio = (alt - base_expanded).norm(dim=-1) / base_expanded.norm(dim=-1).clamp_min(1e-12)
        valid = mask.unsqueeze(1).expand_as(cosine)
        cosine_valid = cosine[valid]
        ratio_valid = delta_ratio[valid]

        stats: Dict[str, object] = {
            "base_alt_cosine_mean": float(cosine_valid.mean().cpu().item()) if cosine_valid.numel() else 0.0,
            "base_alt_cosine_std": float(cosine_valid.std(unbiased=False).cpu().item()) if cosine_valid.numel() else 0.0,
            "base_alt_l2_ratio_mean": float(ratio_valid.mean().cpu().item()) if ratio_valid.numel() else 0.0,
            "base_alt_l2_ratio_std": float(ratio_valid.std(unbiased=False).cpu().item()) if ratio_valid.numel() else 0.0,
        }

        if alt.shape[1] > 1:
            alt_flat = alt.transpose(1, 2)  # (B, S, P, D)
            alt_norm = F.normalize(alt_flat, dim=-1, eps=1e-12)
            pairwise = alt_norm @ alt_norm.transpose(-1, -2)
            p = alt.shape[1]
            eye = torch.eye(p, device=pairwise.device, dtype=torch.bool)
            pairwise_offdiag = pairwise[..., ~eye]
            valid_pairwise = pairwise_offdiag[mask]
            stats["alt_alt_cosine_mean"] = (
                float(valid_pairwise.mean().cpu().item()) if valid_pairwise.numel() else 0.0
            )
            stats["alt_alt_cosine_std"] = (
                float(valid_pairwise.std(unbiased=False).cpu().item()) if valid_pairwise.numel() else 0.0
            )
        else:
            stats["alt_alt_cosine_mean"] = None
            stats["alt_alt_cosine_std"] = None
        return stats

    @staticmethod
    def _prediction_stats_from_nll_and_top1(
        nll_by_traj: torch.Tensor,
        top1_by_traj: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, object]:
        if nll_by_traj.ndim != 3:
            raise ValueError("nll_by_traj must have shape (batch, trajectories, seq_minus_1)")
        if top1_by_traj.shape != nll_by_traj.shape:
            raise ValueError("top1_by_traj must have the same shape as nll_by_traj")
        batch, num_trajectories, seq_minus_1 = nll_by_traj.shape
        if tuple(valid_mask.shape) != (batch, seq_minus_1):
            raise ValueError(
                f"valid_mask must have shape {(batch, seq_minus_1)}, got {tuple(valid_mask.shape)}"
            )

        valid = valid_mask.to(dtype=torch.bool, device=nll_by_traj.device)
        if not bool(valid.any().item()):
            zeros = [0.0 for _ in range(num_trajectories)]
            return {
                "ce_by_traj": zeros,
                "ce_delta_vs_base_by_traj": zeros,
                "oracle_ce": 0.0,
                "oracle_ce_delta_vs_base": 0.0,
                "gold_nll_std_mean": 0.0,
                "gold_nll_range_mean": 0.0,
                "top1_match_with_base": [0.0 for _ in range(num_trajectories - 1)],
            }

        nll_valid = nll_by_traj.float().permute(0, 2, 1)[valid]
        ce_by_traj_tensor = nll_valid.mean(dim=0)
        ce_by_traj = ce_by_traj_tensor.cpu().tolist()
        ce_delta = (ce_by_traj_tensor - ce_by_traj_tensor[0]).cpu().tolist()
        oracle_nll = nll_valid.min(dim=-1).values
        nll_range = nll_valid.max(dim=-1).values - nll_valid.min(dim=-1).values

        top1_valid = top1_by_traj.permute(0, 2, 1)[valid]
        if num_trajectories > 1:
            top1_match = (
                top1_valid[:, 1:] == top1_valid[:, 0:1]
            ).float().mean(dim=0).cpu().tolist()
        else:
            top1_match = []

        return {
            "ce_by_traj": ce_by_traj,
            "ce_delta_vs_base_by_traj": ce_delta,
            "oracle_ce": float(oracle_nll.mean().cpu().item()),
            "oracle_ce_delta_vs_base": float((oracle_nll.mean() - ce_by_traj_tensor[0]).cpu().item()),
            "gold_nll_std_mean": float(nll_valid.std(dim=-1, unbiased=False).mean().cpu().item()),
            "gold_nll_range_mean": float(nll_range.mean().cpu().item()),
            "top1_match_with_base": top1_match,
        }

    @staticmethod
    def _aggregator_alignment_stats_from_nll_and_alpha(
        nll_by_traj: torch.Tensor,
        aggregator_alpha: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, object]:
        if nll_by_traj.ndim != 3:
            raise ValueError("nll_by_traj must have shape (batch, trajectories, seq_minus_1)")
        if aggregator_alpha.ndim != 3:
            raise ValueError("aggregator_alpha must have shape (batch, candidates, seq)")
        batch, num_trajectories, seq_minus_1 = nll_by_traj.shape
        if tuple(valid_mask.shape) != (batch, seq_minus_1):
            raise ValueError(
                f"valid_mask must have shape {(batch, seq_minus_1)}, got {tuple(valid_mask.shape)}"
            )
        expected_alt = num_trajectories - 1
        if aggregator_alpha.shape[0] != batch:
            raise ValueError("aggregator_alpha batch dimension does not match nll_by_traj")
        if aggregator_alpha.shape[-1] == seq_minus_1 + 1:
            alpha = aggregator_alpha[:, :, :-1]
        elif aggregator_alpha.shape[-1] == seq_minus_1:
            alpha = aggregator_alpha
        else:
            raise ValueError(
                f"aggregator_alpha seq dimension must be {seq_minus_1} or {seq_minus_1 + 1}, "
                f"got {aggregator_alpha.shape[-1]}"
            )
        if alpha.shape[1] not in (expected_alt, expected_alt + 1):
            raise ValueError(
                f"aggregator_alpha candidates must be {expected_alt} or {expected_alt + 1}, "
                f"got {alpha.shape[1]}"
            )

        valid = valid_mask.to(dtype=torch.bool, device=nll_by_traj.device)
        if not bool(valid.any().item()):
            return {
                "aggregator_alpha_on_best_alt_mean": 0.0,
                "aggregator_alt_oracle_regret_mean": 0.0,
                "aggregator_alpha_best_alt_top1_match": None,
                "aggregator_alpha_oracle_corr_mean": None,
                "aggregator_alt_better_rate": 0.0,
                "aggregator_alt_mass_mean_on_alt_better": 0.0,
                "aggregator_alt_mass_mean_on_base_better": 0.0,
                "aggregator_null_alpha_mean_on_alt_better": None,
                "aggregator_null_alpha_mean_on_base_better": None,
            }

        nll = nll_by_traj.float()
        alpha = alpha.to(device=nll.device, dtype=torch.float32)
        alt_nll = nll[:, 1:]
        base_nll = nll[:, 0]
        alt_alpha = alpha[:, :expected_alt]
        null_alpha = alpha[:, expected_alt] if alpha.shape[1] == expected_alt + 1 else None

        alt_nll_valid = alt_nll.permute(0, 2, 1)[valid]
        alt_alpha_valid = alt_alpha.permute(0, 2, 1)[valid]
        base_nll_valid = base_nll[valid]
        alt_min, best_alt_idx = alt_nll_valid.min(dim=-1)
        alpha_on_best = alt_alpha_valid.gather(dim=-1, index=best_alt_idx.unsqueeze(-1)).squeeze(-1)
        alt_mass = alt_alpha_valid.sum(dim=-1)
        normalized_alt_alpha = alt_alpha_valid / alt_mass.clamp_min(1e-12).unsqueeze(-1)
        weighted_alt_nll = (normalized_alt_alpha * alt_nll_valid).sum(dim=-1)
        regret = weighted_alt_nll - alt_min
        alt_better = alt_min < base_nll_valid
        base_better = ~alt_better

        def mean_when(values: torch.Tensor, mask: torch.Tensor) -> float:
            if bool(mask.any().item()):
                return float(values[mask].mean().cpu().item())
            return 0.0

        if expected_alt > 1:
            alpha_argmax = alt_alpha_valid.argmax(dim=-1)
            best_match = (alpha_argmax == best_alt_idx).float().mean()
            alpha_centered = alt_alpha_valid - alt_alpha_valid.mean(dim=-1, keepdim=True)
            utility = -alt_nll_valid
            utility_centered = utility - utility.mean(dim=-1, keepdim=True)
            numerator = (alpha_centered * utility_centered).sum(dim=-1)
            denominator = alpha_centered.pow(2).sum(dim=-1).sqrt() * utility_centered.pow(2).sum(dim=-1).sqrt()
            corr_valid = denominator > 1e-12
            corr = numerator[corr_valid] / denominator[corr_valid]
            corr_value = float(corr.mean().cpu().item()) if corr.numel() else None
            best_match_value: Optional[float] = float(best_match.cpu().item())
        else:
            best_match_value = None
            corr_value = None

        if null_alpha is not None:
            null_valid = null_alpha[:, :seq_minus_1][valid]
            null_alt = mean_when(null_valid, alt_better)
            null_base = mean_when(null_valid, base_better)
        else:
            null_alt = None
            null_base = None

        return {
            "aggregator_alpha_on_best_alt_mean": float(alpha_on_best.mean().cpu().item()),
            "aggregator_alt_oracle_regret_mean": float(regret.mean().cpu().item()),
            "aggregator_alpha_best_alt_top1_match": best_match_value,
            "aggregator_alpha_oracle_corr_mean": corr_value,
            "aggregator_alt_better_rate": float(alt_better.float().mean().cpu().item()),
            "aggregator_alt_mass_mean_on_alt_better": mean_when(alt_mass, alt_better),
            "aggregator_alt_mass_mean_on_base_better": mean_when(alt_mass, base_better),
            "aggregator_null_alpha_mean_on_alt_better": null_alt,
            "aggregator_null_alpha_mean_on_base_better": null_base,
        }

    @staticmethod
    def _gold_nll_from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = labels.to(logits.device)
        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid = shift_labels.ne(-100)
        safe_labels = shift_labels.clamp_min(0)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        return nll.detach().float().cpu(), valid.detach().cpu()

    @staticmethod
    def _fusion_improvement_stats_from_nll(
        nll_by_traj: torch.Tensor,
        fused_nll: torch.Tensor,
        base_nll: torch.Tensor,
        valid_mask: torch.Tensor,
        aggregator_alpha: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if nll_by_traj.ndim != 3:
            raise ValueError("nll_by_traj must have shape (batch, trajectories, seq_minus_1)")
        batch, num_trajectories, seq_minus_1 = nll_by_traj.shape
        expected_shape = (batch, seq_minus_1)
        if tuple(fused_nll.shape) != expected_shape:
            raise ValueError(f"fused_nll must have shape {expected_shape}, got {tuple(fused_nll.shape)}")
        if tuple(base_nll.shape) != expected_shape:
            raise ValueError(f"base_nll must have shape {expected_shape}, got {tuple(base_nll.shape)}")
        if tuple(valid_mask.shape) != expected_shape:
            raise ValueError(f"valid_mask must have shape {expected_shape}, got {tuple(valid_mask.shape)}")

        valid = valid_mask.to(dtype=torch.bool, device=nll_by_traj.device)
        if not bool(valid.any().item()):
            return {
                "fusion_improvement_mean": 0.0,
                "fusion_beats_base_rate": 0.0,
                "fusion_regret_vs_oracle_mean": 0.0,
                "fusion_regret_vs_alt_oracle_mean": 0.0,
                "alt_advantage_mean": 0.0,
                "alt_advantage_positive_rate": 0.0,
                "fusion_improvement_on_alt_better": 0.0,
                "fusion_improvement_on_base_better": 0.0,
                "fusion_improvement_alt_advantage_corr": None,
                "aggregator_alt_mass_alt_advantage_corr": None,
            }

        nll = nll_by_traj.float().to(device=fused_nll.device)
        fused = fused_nll.float()
        base = base_nll.float()
        valid = valid.to(device=fused.device)
        all_oracle = nll.min(dim=1).values
        alt_oracle = nll[:, 1:].min(dim=1).values
        improvement = base - fused
        alt_advantage = base - alt_oracle
        fusion_regret = fused - all_oracle
        fusion_regret_alt = fused - alt_oracle
        alt_better = alt_advantage > 0
        base_better = ~alt_better

        improvement_valid = improvement[valid]
        alt_advantage_valid = alt_advantage[valid]
        alt_better_valid = alt_better[valid]
        base_better_valid = base_better[valid]

        def mean_when(values: torch.Tensor, mask: torch.Tensor) -> float:
            if bool(mask.any().item()):
                return float(values[mask].mean().cpu().item())
            return 0.0

        def corr_or_none(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
            if x.numel() < 2:
                return None
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            denom = x_centered.pow(2).sum().sqrt() * y_centered.pow(2).sum().sqrt()
            if float(denom.cpu().item()) <= 1e-12:
                return None
            return float((x_centered * y_centered).sum().div(denom).cpu().item())

        alt_mass_corr = None
        if aggregator_alpha is not None:
            alpha = aggregator_alpha.detach().float().cpu()
            if alpha.shape[0] != batch:
                raise ValueError("aggregator_alpha batch dimension does not match nll_by_traj")
            if alpha.shape[-1] == seq_minus_1 + 1:
                alpha = alpha[:, :, :-1]
            elif alpha.shape[-1] != seq_minus_1:
                raise ValueError(
                    f"aggregator_alpha seq dimension must be {seq_minus_1} or {seq_minus_1 + 1}, "
                    f"got {alpha.shape[-1]}"
                )
            expected_alt = num_trajectories - 1
            if alpha.shape[1] not in (expected_alt, expected_alt + 1):
                raise ValueError(
                    f"aggregator_alpha candidates must be {expected_alt} or {expected_alt + 1}, "
                    f"got {alpha.shape[1]}"
                )
            alt_mass = alpha[:, :expected_alt].sum(dim=1).to(device=alt_advantage.device)
            alt_mass_corr = corr_or_none(alt_mass[valid], alt_advantage_valid)

        return {
            "fusion_improvement_mean": float(improvement_valid.mean().cpu().item()),
            "fusion_beats_base_rate": float((fused[valid] < base[valid]).float().mean().cpu().item()),
            "fusion_regret_vs_oracle_mean": float(fusion_regret[valid].mean().cpu().item()),
            "fusion_regret_vs_alt_oracle_mean": float(fusion_regret_alt[valid].mean().cpu().item()),
            "alt_advantage_mean": float(alt_advantage_valid.mean().cpu().item()),
            "alt_advantage_positive_rate": float(alt_better_valid.float().mean().cpu().item()),
            "fusion_improvement_on_alt_better": mean_when(improvement_valid, alt_better_valid),
            "fusion_improvement_on_base_better": mean_when(improvement_valid, base_better_valid),
            "fusion_improvement_alt_advantage_corr": corr_or_none(
                improvement_valid,
                alt_advantage_valid,
            ),
            "aggregator_alt_mass_alt_advantage_corr": alt_mass_corr,
        }

    @torch.no_grad()
    def _trajectory_prediction_stats(
        self,
        pre_norm_by_traj: torch.Tensor,
        labels: torch.Tensor,
        aggregator_alpha: Optional[torch.Tensor] = None,
        fused_logits: Optional[torch.Tensor] = None,
        base_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        nll_by_traj, top1_by_traj, valid = self._trajectory_nll_and_top1(
            pre_norm_by_traj=pre_norm_by_traj,
            labels=labels,
            detach_to_cpu=True,
        )
        stats = self._prediction_stats_from_nll_and_top1(
            nll_by_traj=nll_by_traj,
            top1_by_traj=top1_by_traj,
            valid_mask=valid,
        )
        if aggregator_alpha is not None:
            stats.update(
                self._aggregator_alignment_stats_from_nll_and_alpha(
                    nll_by_traj=nll_by_traj,
                    aggregator_alpha=aggregator_alpha.detach().float().cpu(),
                    valid_mask=valid,
                )
            )
        if fused_logits is not None and base_logits is not None:
            fused_nll, fused_valid = self._gold_nll_from_logits(fused_logits, labels)
            base_nll, base_valid = self._gold_nll_from_logits(base_logits, labels)
            if tuple(fused_valid.shape) != tuple(valid.shape) or tuple(base_valid.shape) != tuple(valid.shape):
                raise ValueError("fused/base valid masks do not match trajectory valid mask")
            stats.update(
                self._fusion_improvement_stats_from_nll(
                    nll_by_traj=nll_by_traj,
                    fused_nll=fused_nll,
                    base_nll=base_nll,
                    valid_mask=valid & fused_valid & base_valid,
                    aggregator_alpha=aggregator_alpha,
                )
            )
        return stats

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        kl_beta: float = 0.0,
        kl_direction: str = "base_to_fused",
        kl_advantage_tau: float = 0.0,
        residual_l2_weight: float = 0.0,
        trajectory_oracle_aux_weight: float = 0.0,
        trajectory_oracle_temperature: float = 0.5,
        trajectory_oracle_include_base: bool = False,
        aggregator_oracle_align_weight: float = 0.0,
        aggregator_oracle_align_temperature: float = 0.5,
        return_base_logits: bool = False,
        return_route_stats: bool = False,
        return_path_stats: bool = False,
        return_aggregator_stats: bool = False,
        return_trajectory_stats: bool = False,
        return_trajectory_prediction_stats: bool = False,
        return_token_routing: bool = False,
        return_pre_norm_hidden: bool = False,
        return_aggregator_alpha: bool = False,
        use_cache: bool = False,
        **_: object,
    ) -> TrajectoryCausalLMOutput:
        if use_cache:
            raise NotImplementedError("TrajectoryEnsembleForCausalLM v1 supports use_cache=False only")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq)")
        if kl_advantage_tau < 0:
            raise ValueError("kl_advantage_tau must be non-negative")

        batch, seq_len = input_ids.shape
        expanded_input_ids = self._expand_by_trajectory(input_ids)
        expanded_attention_mask = (
            self._expand_by_trajectory(attention_mask) if attention_mask is not None else None
        )
        if position_ids is not None and position_ids.shape[0] == 1 and batch > 1:
            position_ids = position_ids.expand(batch, -1)
        expanded_position_ids = self._expand_by_trajectory(position_ids) if position_ids is not None else None

        previous_recording = self.router_noise.record_routing
        previous_token_recording = self.router_noise.record_token_routing
        previous_record_mask = self.router_noise.record_mask
        previous_current_mask = self.router_noise.current_mask
        self.router_noise.clear_stats()
        self.router_noise.record_routing = return_route_stats or return_path_stats
        self.router_noise.record_token_routing = return_token_routing
        if attention_mask is not None:
            self.router_noise.current_mask = attention_mask.detach().bool()
        if self.router_noise.record_routing and attention_mask is not None:
            self.router_noise.record_mask = attention_mask.detach().bool()
        try:
            pre_norm = self._run_deepseek_pre_norm(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                position_ids=expanded_position_ids,
            )
        finally:
            self.router_noise.record_routing = previous_recording
            self.router_noise.record_token_routing = previous_token_recording
            self.router_noise.record_mask = previous_record_mask
            self.router_noise.current_mask = previous_current_mask

        pre_norm = pre_norm.reshape(batch, self.num_trajectories, seq_len, -1)
        need_aggregator_alpha = return_aggregator_alpha or (
            labels is not None
            and (return_trajectory_prediction_stats or aggregator_oracle_align_weight > 0)
        )
        aggregator_alpha = None
        if need_aggregator_alpha:
            fused_pre_norm, aggregator_stats, aggregator_alpha = self.aggregator(
                pre_norm,
                attention_mask=attention_mask,
                return_stats=return_aggregator_stats,
                return_alpha=True,
            )
        else:
            fused_pre_norm, aggregator_stats = self.aggregator(
                pre_norm,
                attention_mask=attention_mask,
                return_stats=return_aggregator_stats,
            )
        trajectory_stats = self._trajectory_stats(pre_norm, attention_mask) if return_trajectory_stats else None
        fused_hidden = self.deepseek_model.norm(fused_pre_norm)
        logits = self.lm_head(fused_hidden).float()

        base_logits = None
        if labels is not None or return_base_logits:
            base_pre_norm = pre_norm[:, 0]
            base_hidden = self.deepseek_model.norm(base_pre_norm)
            base_logits = self.lm_head(base_hidden).float()

        trajectory_prediction_stats = (
            self._trajectory_prediction_stats(
                pre_norm,
                labels,
                aggregator_alpha=aggregator_alpha,
                fused_logits=logits,
                base_logits=base_logits,
            )
            if return_trajectory_prediction_stats and labels is not None
            else None
        )

        loss = None
        components = None
        if labels is not None:
            kl_token_weight = None
            if kl_beta > 0 and kl_advantage_tau > 0:
                # Selective preservation: keep KL strong where the base route
                # already wins and relax it where alternatives carry better
                # gold-token evidence. Needs per-trajectory LM-head NLLs.
                with torch.no_grad():
                    nll_by_traj, _, _ = self._trajectory_nll_and_top1(
                        pre_norm_by_traj=pre_norm,
                        labels=labels,
                        detach_to_cpu=False,
                    )
                    alt_advantage = (
                        nll_by_traj[:, 0] - nll_by_traj[:, 1:].min(dim=1).values
                    )
                    kl_token_weight = torch.sigmoid(
                        -alt_advantage / float(kl_advantage_tau)
                    )
            residual_l2 = self._masked_residual_l2_ratio(
                fused_pre_norm=fused_pre_norm,
                base_pre_norm=pre_norm[:, 0],
                attention_mask=attention_mask,
            )
            trajectory_oracle_aux_loss = None
            trajectory_oracle_aux_components = None
            if trajectory_oracle_aux_weight > 0:
                trajectory_oracle_aux_loss, trajectory_oracle_aux_components = (
                    self._trajectory_oracle_auxiliary_loss(
                        pre_norm_by_traj=pre_norm,
                        labels=labels,
                        temperature=trajectory_oracle_temperature,
                        include_base=trajectory_oracle_include_base,
                    )
                )
            aggregator_oracle_align_loss = None
            aggregator_oracle_align_components = None
            if aggregator_oracle_align_weight > 0:
                if aggregator_alpha is None:
                    raise RuntimeError("aggregator_alpha was not computed for aggregator oracle alignment")
                aggregator_oracle_align_loss, aggregator_oracle_align_components = (
                    self._aggregator_oracle_alignment_loss(
                        pre_norm_by_traj=pre_norm,
                        labels=labels,
                        aggregator_alpha=aggregator_alpha,
                        temperature=aggregator_oracle_align_temperature,
                    )
                )
            loss, components = self._compute_loss(
                logits=logits,
                labels=labels,
                base_logits=base_logits,
                kl_beta=kl_beta,
                kl_direction=kl_direction,
                kl_token_weight=kl_token_weight,
                residual_l2=residual_l2,
                residual_l2_weight=residual_l2_weight,
                trajectory_oracle_aux_loss=trajectory_oracle_aux_loss,
                trajectory_oracle_aux_weight=trajectory_oracle_aux_weight,
                trajectory_oracle_aux_components=trajectory_oracle_aux_components,
                aggregator_oracle_align_loss=aggregator_oracle_align_loss,
                aggregator_oracle_align_weight=aggregator_oracle_align_weight,
                aggregator_oracle_align_components=aggregator_oracle_align_components,
            )
            if kl_token_weight is not None:
                components["kl_advantage_tau"] = float(kl_advantage_tau)

        return TrajectoryCausalLMOutput(
            loss=loss,
            logits=logits,
            base_logits=base_logits if return_base_logits or labels is not None else None,
            route_stats=self.router_noise.stats_as_dict() if return_route_stats else None,
            path_stats=(
                self.router_noise.path_stats_as_dict()
                if return_route_stats or return_path_stats
                else None
            ),
            aggregator_stats=aggregator_stats.as_dict() if aggregator_stats is not None else None,
            trajectory_stats=trajectory_stats,
            trajectory_prediction_stats=trajectory_prediction_stats,
            loss_components=components,
            token_routing=dict(self.router_noise.token_routing) if return_token_routing else None,
            pre_norm_by_traj=pre_norm.detach() if return_pre_norm_hidden else None,
            aggregator_alpha=(
                aggregator_alpha.detach() if return_aggregator_alpha and aggregator_alpha is not None else None
            ),
        )

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        was_training = self.training
        self.eval()
        input_ids = input_ids.to(self.input_device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.input_device)
        else:
            attention_mask = attention_mask.to(self.input_device)

        generated = input_ids
        attn = attention_mask
        for _ in range(max_new_tokens):
            output = self.forward(generated, attention_mask=attn)
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True).to(generated.device)
            generated = torch.cat([generated, next_token], dim=-1)
            attn = torch.cat([attn, torch.ones_like(next_token, device=attn.device)], dim=-1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
                break
        if was_training:
            self.train(True)
        return generated
