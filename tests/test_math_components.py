from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aggregator import BaseAnchoredResidualAggregator
from src.modeling_trajectory import TrajectoryEnsembleForCausalLM
from src.router_noise import (
    SeedRouterNoise,
    _summarize_routing,
    _update_path_stats,
    patch_deepseek_moe_gates,
)

_TRAIN_ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "train_adapter_for_tests",
    ROOT / "scripts" / "train_adapter.py",
)
assert _TRAIN_ADAPTER_SPEC is not None
_TRAIN_ADAPTER = importlib.util.module_from_spec(_TRAIN_ADAPTER_SPEC)
assert _TRAIN_ADAPTER_SPEC.loader is not None
sys.modules[_TRAIN_ADAPTER_SPEC.name] = _TRAIN_ADAPTER
_TRAIN_ADAPTER_SPEC.loader.exec_module(_TRAIN_ADAPTER)
TrainingExample = _TRAIN_ADAPTER.TrainingExample
encode_batch = _TRAIN_ADAPTER.encode_batch
grad_summary = _TRAIN_ADAPTER.grad_summary
freeze_seed_noise = _TRAIN_ADAPTER.freeze_seed_noise


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DECONTAM = _load_script_module("decontamination_for_tests", "decontamination.py")
_EVALUATE = _load_script_module("evaluate_for_tests", "evaluate.py")
_DATA_QUALITY = _load_script_module("data_quality_for_tests", "data_quality.py")
_MINER = _load_script_module("mine_hard_prefixes_for_tests", "mine_hard_prefixes.py")


class ToyEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

    def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False):
        del truncation
        pieces = text.split()
        ids = [1] if add_special_tokens else []
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            ids.append(self.vocab[piece])
        return ToyEncoding(ids)

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool = False):
        del tokenize
        rendered = " ".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered = rendered + " assistant:"
        return rendered.strip()


class TinyExpert(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MoEGate(nn.Module):
    def __init__(self, hidden_size: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = n_experts
        self.scoring_func = "softmax"
        self.norm_topk_prob = False
        self.weight = nn.Parameter(torch.randn(n_experts, hidden_size) * 0.1)

    def forward(self, hidden_states: torch.Tensor):
        raise AssertionError("patch_deepseek_moe_gates should replace this method")


class DeepseekMoE(nn.Module):
    def __init__(self, hidden_size: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_shared_experts=None)
        self.gate = MoEGate(hidden_size, n_experts, top_k)
        self.experts = nn.ModuleList([TinyExpert(hidden_size) for _ in range(n_experts)])
        self.num_experts_per_tok = top_k

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise AssertionError("patch_deepseek_moe_gates should replace this method")


class DenseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, **_: object):
        return (self.identity(hidden_states),)


class TinyLayer(nn.Module):
    def __init__(self, hidden_size: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.mlp = DeepseekMoE(hidden_size, n_experts, top_k)

    def forward(self, hidden_states: torch.Tensor, **_: object):
        return (self.mlp(hidden_states),)


class TinyDeepSeek(nn.Module):
    def __init__(self, hidden_size: int = 8, n_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                DenseLayer(),
                TinyLayer(hidden_size, n_experts, top_k),
                TinyLayer(hidden_size, n_experts, top_k),
            ]
        )


class TinyBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 8,
        n_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self._use_flash_attention_2 = True
        self._use_sdpa = False
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                DenseLayer(),
                TinyLayer(hidden_size, n_experts, top_k),
                TinyLayer(hidden_size, n_experts, top_k),
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 8,
        n_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            n_routed_experts=n_experts,
        )
        self.model = TinyBackbone(vocab_size, hidden_size, n_experts, top_k)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


def test_aggregator_identity_and_initial_gradient() -> None:
    torch.manual_seed(0)
    agg = BaseAnchoredResidualAggregator(hidden_size=8, agg_dim=4)
    hidden = torch.randn(2, 3, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    fused, stats, alpha = agg(hidden, attention_mask=mask, return_stats=True, return_alpha=True)

    assert torch.equal(fused, hidden[:, 0])
    assert stats is not None
    assert alpha.shape == (2, 3, 5)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2, 5), atol=1e-6)
    assert stats.alpha_mean.shape == (3,)
    assert torch.allclose(stats.alpha_mean.sum(), torch.tensor(1.0), atol=1e-6)
    assert stats.alpha_entropy_mean >= 0.0
    assert 0.0 <= stats.alpha_max_mean <= 1.0
    assert stats.null_alpha_mean is not None
    assert 0.0 <= stats.null_alpha_mean <= 1.0
    assert torch.isclose(
        torch.tensor(stats.alt_alpha_mass_mean + stats.null_alpha_mean),
        torch.tensor(1.0),
        atol=1e-6,
    )
    assert stats.residual_norm_ratio == 0.0
    assert 0.0 < stats.residual_scale < 0.25

    loss = fused.float().pow(2).mean()
    loss.backward()
    assert agg.out_proj.weight.grad is not None
    assert agg.out_proj.weight.grad.abs().sum() > 0


def test_aggregator_can_disable_null_candidate() -> None:
    torch.manual_seed(11)
    agg = BaseAnchoredResidualAggregator(hidden_size=8, agg_dim=4, include_null_candidate=False)
    hidden = torch.randn(2, 3, 5, 8)
    fused, stats = agg(hidden, return_stats=True)
    assert torch.equal(fused, hidden[:, 0])
    assert stats is not None
    assert stats.alpha_mean.shape == (2,)
    assert stats.null_alpha_mean is None
    assert torch.isclose(torch.tensor(stats.alt_alpha_mass_mean), torch.tensor(1.0), atol=1e-6)


def test_aggregator_delta_value_ignores_identical_alt_hidden() -> None:
    torch.manual_seed(12)
    base = torch.randn(2, 5, 8)
    hidden = torch.stack([base, base.clone(), base.clone()], dim=1)
    delta_agg = BaseAnchoredResidualAggregator(
        hidden_size=8,
        agg_dim=4,
        include_null_candidate=False,
        value_mode="delta",
    )
    fused = delta_agg(hidden)[0]
    fused.float().pow(2).mean().backward()
    assert torch.equal(fused, base)
    assert delta_agg.out_proj.weight.grad is not None
    assert delta_agg.out_proj.weight.grad.abs().sum() == 0

    absolute_agg = BaseAnchoredResidualAggregator(
        hidden_size=8,
        agg_dim=4,
        include_null_candidate=False,
        value_mode="absolute",
    )
    fused_abs = absolute_agg(hidden)[0]
    fused_abs.float().pow(2).mean().backward()
    assert absolute_agg.out_proj.weight.grad is not None
    assert absolute_agg.out_proj.weight.grad.abs().sum() > 0


def test_orthogonal_seed_init_is_centered_orthogonal_and_scaled() -> None:
    torch.manual_seed(21)
    noise = SeedRouterNoise(num_trajectories=5, n_experts=64, noise_init_std=0.02)
    assert noise.seed_init_mode == "orthogonal"
    rows = noise.noise.detach()
    assert torch.allclose(rows.mean(dim=-1), torch.zeros(4), atol=1e-6)
    gram = rows @ rows.T
    off_diag = gram - torch.diag(torch.diag(gram))
    assert float(off_diag.abs().max()) < 1e-5
    expected_norm = 0.02 * (64**0.5)
    assert torch.allclose(rows.norm(dim=-1), torch.full((4,), expected_norm), atol=1e-5)

    gaussian = SeedRouterNoise(
        num_trajectories=3, n_experts=8, noise_init_std=0.02, seed_init_mode="gaussian"
    )
    assert gaussian.seed_init_mode == "gaussian"

    # Toy regime where the zero-sum subspace is smaller than the row count:
    # init must stay finite and centered instead of failing.
    tiny = SeedRouterNoise(num_trajectories=5, n_experts=4, noise_init_std=0.02)
    assert torch.isfinite(tiny.noise).all()
    assert torch.allclose(tiny.noise.mean(dim=-1), torch.zeros(4), atol=1e-5)


def test_relative_keys_center_scores_and_keep_identity_init() -> None:
    torch.manual_seed(22)
    agg = BaseAnchoredResidualAggregator(hidden_size=8, agg_dim=4, relative_keys=True)
    hidden = torch.randn(2, 3, 5, 8)
    fused, _, alpha = agg(hidden, return_stats=False, return_alpha=True)
    assert torch.equal(fused, hidden[:, 0])

    base_normed = agg.norm(hidden[:, 0])
    q = agg.q_proj(base_normed) + agg.global_q_proj(base_normed.mean(dim=1)).unsqueeze(1)
    k = agg.k_proj(agg.norm(hidden[:, 1:]))
    k = k - k.mean(dim=1, keepdim=True)
    scores = (q.unsqueeze(1) * k).sum(dim=-1) / (4**0.5)
    null_scores = (q * agg.null_key.view(1, 1, -1)).sum(dim=-1, keepdim=True)
    null_scores = null_scores.transpose(1, 2) / (4**0.5)
    expected_alpha = torch.softmax(torch.cat([scores, null_scores], dim=1), dim=1)
    assert torch.allclose(alpha, expected_alpha, atol=1e-6)
    # Centered keys make the two alt scores sum to zero at every token.
    assert torch.allclose(scores.sum(dim=1), torch.zeros(2, 5), atol=1e-5)


def test_first_router_noise_is_centered_and_base_is_zero() -> None:
    noise = SeedRouterNoise(num_trajectories=3, n_experts=4, noise_scale=0.5)
    assert torch.allclose(noise.noise_scale_value().view(-1), torch.full((2,), 0.5), atol=1e-6)
    bias = noise.full_bias(device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(bias[0], torch.zeros_like(bias[0]))
    assert torch.allclose(bias[1:].mean(dim=-1), torch.zeros(2), atol=1e-6)
    reg = noise.diversity_loss() + 1e-4 * noise.l2_loss()
    reg.backward()
    assert noise.noise.grad is not None
    assert noise.noise.grad.abs().sum() > 0
    assert noise.raw_noise_scale.grad is not None


def test_disable_seed_noise_is_true_ablation() -> None:
    noise = SeedRouterNoise(num_trajectories=3, n_experts=4, noise_scale=0.5, disable_noise=True)
    noise.target_layer_idx = 1
    noise.train(True)
    bias = noise.full_bias(device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(bias, torch.zeros_like(bias))
    assert noise.should_inject(1) is False
    assert noise.use_relaxed_training_router(1) is False


def test_context_seed_gate_is_identity_initialized_and_bounded() -> None:
    torch.manual_seed(9)
    noise = SeedRouterNoise(
        num_trajectories=3,
        n_experts=4,
        noise_scale=0.5,
        hidden_size=8,
        context_seed_gate=True,
        context_scale_max_delta=0.5,
    )
    hidden = torch.randn(6, 3, 8)
    bias, scale = noise.bias_for_hidden_states(hidden, dtype=torch.float32)
    expected = noise.full_bias(device=torch.device("cpu"), dtype=torch.float32).repeat(2, 1)
    assert bias is not None
    assert torch.allclose(bias, expected, atol=1e-6)
    assert scale == [0.0, 0.5, 0.5]

    with torch.no_grad():
        noise.context_scale_proj.bias.copy_(torch.tensor([10.0, -10.0]))
    _, shifted_scale = noise.bias_for_hidden_states(hidden, dtype=torch.float32)
    assert shifted_scale[1] <= 0.75 + 1e-6
    assert shifted_scale[2] >= 0.25 - 1e-6
    assert shifted_scale[1] > shifted_scale[2]


def test_context_seed_gate_gets_gradient_through_moe_output() -> None:
    torch.manual_seed(10)
    tiny = TinyDeepSeek()
    noise = SeedRouterNoise(
        num_trajectories=3,
        n_experts=4,
        noise_scale=0.8,
        hidden_size=8,
        context_seed_gate=True,
    )
    patch_deepseek_moe_gates(tiny, noise)
    hidden = torch.randn(3, 2, 8)
    output = tiny.model.layers[1].mlp(hidden)
    output.float().pow(2).mean().backward()
    assert noise.context_scale_proj.weight.grad is not None
    assert noise.context_scale_proj.weight.grad.abs().sum() > 0


def test_first_router_noise_gets_gradient_through_moe_output() -> None:
    torch.manual_seed(1)
    tiny = TinyDeepSeek()
    noise = SeedRouterNoise(num_trajectories=3, n_experts=4, noise_scale=0.8)
    patched_layers = patch_deepseek_moe_gates(tiny, noise)
    assert patched_layers == [1, 2]
    assert noise.target_layer_idx == 1

    hidden = torch.randn(3, 2, 8)
    noise.record_routing = True
    first_out = tiny.model.layers[1].mlp(hidden)
    second_out = tiny.model.layers[2].mlp(first_out)
    loss = second_out.float().pow(2).mean()
    loss.backward()

    assert noise.noise.grad is not None
    assert noise.noise.grad.abs().sum() > 0
    assert noise.layer_stats[1].noise_applied is True
    assert noise.layer_stats[2].noise_applied is False
    assert len(noise.layer_stats[1].topk_overlap_with_base) == 2
    assert len(noise.layer_stats[1].topk_exact_match_with_base) == 2
    assert len(noise.layer_stats[1].topk_margin_by_traj) == 3
    assert len(noise.layer_stats[1].topk_logit_margin_by_traj) == 3
    assert len(noise.layer_stats[1].expert_jsd_with_base) == 2
    assert min(noise.layer_stats[1].topk_margin_by_traj) >= 0.0
    assert min(noise.layer_stats[1].topk_logit_margin_by_traj) >= 0.0
    assert min(noise.layer_stats[1].expert_jsd_with_base) >= 0.0
    assert 1 in noise.path_stats
    assert 2 in noise.path_stats
    assert len(noise.path_stats[1].prefix_exact_match_with_base) == 2


def test_first_router_st_topk_forward_matches_hard_eval_but_keeps_dense_gradient() -> None:
    torch.manual_seed(3)
    tiny = TinyDeepSeek()
    noise = SeedRouterNoise(
        num_trajectories=3,
        n_experts=4,
        noise_scale=0.8,
        train_router_mode="st_topk",
    )
    patch_deepseek_moe_gates(tiny, noise)
    hidden = torch.randn(3, 2, 8)

    noise.eval()
    hard_output = tiny.model.layers[1].mlp(hidden)

    noise.train(True)
    train_idx, train_weight, _ = tiny.model.layers[1].mlp.gate(hidden)
    assert train_idx.shape[-1] == 4
    assert torch.count_nonzero(train_weight.detach().abs() > 0).item() <= hidden.shape[0] * hidden.shape[1] * 2
    noise.eval()
    hard_idx, hard_weight, _ = tiny.model.layers[1].mlp.gate(hidden)
    hard_full = torch.zeros_like(train_weight)
    hard_full.scatter_add_(dim=-1, index=hard_idx, src=hard_weight)
    assert torch.allclose(train_weight.detach(), hard_full, atol=1e-6)
    noise.train(True)
    st_output = tiny.model.layers[1].mlp(hidden)
    assert torch.allclose(st_output, hard_output, atol=1e-6)
    st_output.float().pow(2).mean().backward()
    assert noise.noise.grad is not None
    assert noise.noise.grad.abs().sum() > 0

    noise.eval()
    eval_idx, eval_weight, _ = tiny.model.layers[1].mlp.gate(hidden)
    assert eval_idx.shape[-1] == 2
    assert eval_weight.shape[-1] == 2


def test_first_router_soft_all_training_surrogate_switches_to_hard_eval() -> None:
    torch.manual_seed(4)
    tiny = TinyDeepSeek()
    noise = SeedRouterNoise(
        num_trajectories=3,
        n_experts=4,
        noise_scale=0.8,
        train_router_mode="soft_all",
    )
    patch_deepseek_moe_gates(tiny, noise)
    hidden = torch.randn(3, 2, 8)

    noise.train(True)
    train_idx, train_weight, _ = tiny.model.layers[1].mlp.gate(hidden)
    assert train_idx.shape[-1] == 4
    assert float(train_weight.sum(dim=-1).max()) <= 1.0 + 1e-6

    noise.eval()
    eval_idx, eval_weight, _ = tiny.model.layers[1].mlp.gate(hidden)
    assert eval_idx.shape[-1] == 2
    assert eval_weight.shape[-1] == 2


def test_sorted_moe_infer_matches_naive_weighted_sum() -> None:
    torch.manual_seed(2)
    tiny = TinyDeepSeek()
    noise = SeedRouterNoise(num_trajectories=3, n_experts=4, noise_scale=0.8)
    patch_deepseek_moe_gates(tiny, noise)

    moe = tiny.model.layers[1].mlp
    hidden = torch.randn(3, 2, 8)
    topk_idx, topk_weight, _ = moe.gate(hidden)
    expected = torch.zeros_like(hidden.reshape(-1, 8))
    flat_hidden = hidden.reshape(-1, 8)
    flat_idx = topk_idx.reshape(-1)
    flat_weight = topk_weight.reshape(-1, 1)
    token_idx = torch.arange(flat_hidden.shape[0]).repeat_interleave(topk_idx.shape[-1])
    for slot, expert_id in enumerate(flat_idx.tolist()):
        token = token_idx[slot]
        expected[token] += moe.experts[expert_id](flat_hidden[token : token + 1]).squeeze(0) * flat_weight[slot]

    actual = moe(hidden).reshape(-1, 8)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_route_stats_ignore_padding_tokens() -> None:
    scores = torch.tensor(
        [
            [[0.70, 0.20, 0.08, 0.02], [0.70, 0.20, 0.08, 0.02]],
            [[0.70, 0.20, 0.08, 0.02], [0.02, 0.08, 0.20, 0.70]],
            [[0.70, 0.20, 0.08, 0.02], [0.02, 0.08, 0.20, 0.70]],
        ],
        dtype=torch.float32,
    ).reshape(6, 4)
    topk_idx = torch.tensor(
        [
            [[0, 1], [0, 1]],
            [[0, 1], [3, 2]],
            [[0, 1], [3, 2]],
        ],
        dtype=torch.long,
    ).reshape(6, 2)
    valid_mask = torch.tensor([[1, 0]], dtype=torch.bool)

    stats = _summarize_routing(
        logits=scores.log(),
        scores=scores,
        topk_idx=topk_idx,
        layer_idx=1,
        noise_applied=True,
        num_trajectories=3,
        n_experts=4,
        batch_rows=3,
        seq_len=2,
        valid_mask=valid_mask,
    )

    assert stats.topk_overlap_with_base == [1.0, 1.0]
    assert stats.topk_exact_match_with_base == [1.0, 1.0]
    assert stats.expert_jsd_with_base == [0.0, 0.0]


def test_context_gate_regularization_is_zero_init_then_positive() -> None:
    noise = SeedRouterNoise(
        num_trajectories=3,
        n_experts=4,
        noise_scale=0.5,
        hidden_size=8,
        context_seed_gate=True,
    )
    assert float(noise.context_gate_l2_loss()) == 0.0
    assert float(noise.context_gate_norm()) == 0.0
    with torch.no_grad():
        noise.context_scale_proj.bias[0] = 0.25
    assert float(noise.context_gate_l2_loss()) > 0.0
    assert float(noise.context_gate_norm()) > 0.0


def test_path_prefix_remembers_earlier_divergence() -> None:
    noise = SeedRouterNoise(num_trajectories=3, n_experts=4, noise_scale=0.5)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    first_layer_idx = torch.tensor(
        [
            [[0, 1]],
            [[2, 3]],
            [[0, 1]],
        ],
        dtype=torch.long,
    ).reshape(3, 2)
    second_layer_idx = torch.tensor(
        [
            [[0, 1]],
            [[0, 1]],
            [[0, 1]],
        ],
        dtype=torch.long,
    ).reshape(3, 2)

    _update_path_stats(noise, first_layer_idx, layer_idx=1, batch_rows=3, seq_len=1, valid_mask=valid_mask)
    _update_path_stats(noise, second_layer_idx, layer_idx=2, batch_rows=3, seq_len=1, valid_mask=valid_mask)

    assert noise.path_stats[1].prefix_exact_match_with_base == [0.0, 1.0]
    assert noise.path_stats[2].layer_exact_match_with_base == [1.0, 1.0]
    assert noise.path_stats[2].prefix_exact_match_with_base == [0.0, 1.0]


def test_trajectory_stats_are_masked_and_finite() -> None:
    torch.manual_seed(5)
    pre_norm = torch.randn(2, 3, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    stats = TrajectoryEnsembleForCausalLM._trajectory_stats(None, pre_norm, mask)
    assert -1.0 <= stats["base_alt_cosine_mean"] <= 1.0
    assert stats["base_alt_l2_ratio_mean"] >= 0.0
    assert -1.0 <= stats["alt_alt_cosine_mean"] <= 1.0


def test_symmetric_kl_has_gradient_to_fused_logits() -> None:
    torch.manual_seed(6)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    base_logits = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    loss, components = TrajectoryEnsembleForCausalLM._compute_loss(
        None,
        logits=logits,
        labels=labels,
        base_logits=base_logits,
        kl_beta=0.5,
        kl_direction="symmetric",
    )
    loss.backward()
    assert components["kl_direction"] == "symmetric"
    assert "base_ce" in components
    assert "ce_delta_vs_base" in components
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_loss_handles_all_ignored_labels_without_nan() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    base_logits = torch.randn(2, 4, 7)
    labels = torch.full((2, 4), -100, dtype=torch.long)
    loss, components = TrajectoryEnsembleForCausalLM._compute_loss(
        None,
        logits=logits,
        labels=labels,
        base_logits=base_logits,
        kl_beta=0.0,
        kl_direction="base_to_fused",
    )
    assert torch.isfinite(loss)
    assert components["ce"] == 0.0
    assert components["base_ce"] == 0.0
    assert components["ce_delta_vs_base"] == 0.0


def test_loss_handles_single_token_sequence_with_kl() -> None:
    torch.manual_seed(8)
    logits = torch.randn(2, 1, 7, requires_grad=True)
    base_logits = torch.randn(2, 1, 7)
    labels = torch.randint(0, 7, (2, 1))
    loss, components = TrajectoryEnsembleForCausalLM._compute_loss(
        None,
        logits=logits,
        labels=labels,
        base_logits=base_logits,
        kl_beta=1.0,
        kl_direction="base_to_fused",
    )
    assert torch.isfinite(loss)
    assert components["ce"] == 0.0
    assert components["kl"] == 0.0


def test_residual_l2_ratio_is_masked() -> None:
    base = torch.ones(2, 3, 4)
    fused = base.clone()
    fused[0, 0] += 1.0
    fused[1, 2] += 10.0
    mask = torch.tensor([[1, 0, 0], [1, 1, 0]])
    penalty = TrajectoryEnsembleForCausalLM._masked_residual_l2_ratio(
        fused_pre_norm=fused,
        base_pre_norm=base,
        attention_mask=mask,
    )
    assert torch.isclose(penalty, torch.tensor(1.0 / 3.0), atol=1e-6)


def test_prediction_stats_capture_oracle_and_top1_diversity() -> None:
    nll = torch.tensor(
        [
            [
                [2.0, 4.0, 100.0],
                [1.0, 5.0, 100.0],
                [3.0, 2.0, 100.0],
            ]
        ]
    )
    top1 = torch.tensor(
        [
            [
                [10, 11, 12],
                [10, 12, 12],
                [13, 11, 12],
            ]
        ]
    )
    valid = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    stats = TrajectoryEnsembleForCausalLM._prediction_stats_from_nll_and_top1(
        nll_by_traj=nll,
        top1_by_traj=top1,
        valid_mask=valid,
    )
    assert stats["ce_by_traj"] == [3.0, 3.0, 2.5]
    assert stats["ce_delta_vs_base_by_traj"] == [0.0, 0.0, -0.5]
    assert stats["oracle_ce"] == 1.5
    assert stats["oracle_ce_delta_vs_base"] == -1.5
    assert stats["top1_match_with_base"] == [0.5, 0.5]


def test_aggregator_alignment_stats_compare_alpha_to_oracle_nll() -> None:
    nll = torch.tensor(
        [
            [
                [2.0, 1.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    alpha = torch.tensor(
        [
            [
                [0.7, 0.1, 0.5],
                [0.1, 0.1, 0.2],
                [0.2, 0.8, 0.3],
            ]
        ]
    )
    valid = torch.tensor([[1, 1]], dtype=torch.bool)
    stats = TrajectoryEnsembleForCausalLM._aggregator_alignment_stats_from_nll_and_alpha(
        nll_by_traj=nll,
        aggregator_alpha=alpha,
        valid_mask=valid,
    )
    assert torch.isclose(
        torch.tensor(stats["aggregator_alpha_on_best_alt_mean"]),
        torch.tensor(0.4),
        atol=1e-6,
    )
    assert torch.isclose(
        torch.tensor(stats["aggregator_alt_oracle_regret_mean"]),
        torch.tensor(0.625),
        atol=1e-6,
    )
    assert stats["aggregator_alpha_best_alt_top1_match"] == 1.0
    assert stats["aggregator_alt_better_rate"] == 0.5
    assert torch.isclose(
        torch.tensor(stats["aggregator_alt_mass_mean_on_alt_better"]),
        torch.tensor(0.8),
        atol=1e-6,
    )
    assert torch.isclose(
        torch.tensor(stats["aggregator_alt_mass_mean_on_base_better"]),
        torch.tensor(0.2),
        atol=1e-6,
    )
    assert torch.isclose(
        torch.tensor(stats["aggregator_null_alpha_mean_on_alt_better"]),
        torch.tensor(0.2),
        atol=1e-6,
    )
    assert torch.isclose(
        torch.tensor(stats["aggregator_null_alpha_mean_on_base_better"]),
        torch.tensor(0.8),
        atol=1e-6,
    )


def test_fusion_improvement_stats_connect_oracle_advantage_to_fused_gain() -> None:
    nll = torch.tensor(
        [
            [
                [2.0, 4.0],
                [1.0, 5.0],
                [3.0, 2.0],
            ]
        ]
    )
    fused_nll = torch.tensor([[1.5, 3.0]])
    base_nll = torch.tensor([[2.0, 4.0]])
    alpha = torch.tensor(
        [
            [
                [0.7, 0.1, 0.5],
                [0.1, 0.1, 0.2],
                [0.2, 0.8, 0.3],
            ]
        ]
    )
    valid = torch.tensor([[1, 1]], dtype=torch.bool)
    stats = TrajectoryEnsembleForCausalLM._fusion_improvement_stats_from_nll(
        nll_by_traj=nll,
        fused_nll=fused_nll,
        base_nll=base_nll,
        valid_mask=valid,
        aggregator_alpha=alpha,
    )
    assert torch.isclose(torch.tensor(stats["fusion_improvement_mean"]), torch.tensor(0.75), atol=1e-6)
    assert stats["fusion_beats_base_rate"] == 1.0
    assert torch.isclose(
        torch.tensor(stats["fusion_regret_vs_oracle_mean"]),
        torch.tensor(0.75),
        atol=1e-6,
    )
    assert torch.isclose(torch.tensor(stats["alt_advantage_mean"]), torch.tensor(1.5), atol=1e-6)
    assert stats["alt_advantage_positive_rate"] == 1.0
    assert torch.isclose(
        torch.tensor(stats["fusion_improvement_on_alt_better"]),
        torch.tensor(0.75),
        atol=1e-6,
    )
    assert torch.isclose(
        torch.tensor(stats["fusion_improvement_alt_advantage_corr"]),
        torch.tensor(1.0),
        atol=1e-6,
    )
    assert torch.isclose(
        torch.tensor(stats["aggregator_alt_mass_alt_advantage_corr"]),
        torch.tensor(-1.0),
        atol=1e-6,
    )


def test_forward_returns_fusion_conversion_stats_with_identity_init() -> None:
    torch.manual_seed(14)
    base = TinyCausalLM()
    model = TrajectoryEnsembleForCausalLM(
        base_model=base,
        num_trajectories=3,
        agg_dim=4,
        noise_scale=0.4,
        top_k=2,
        context_seed_gate=True,
    )
    input_ids = torch.tensor(
        [
            [1, 3, 4, 5],
            [1, 6, 7, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        kl_beta=1.0,
        return_base_logits=True,
        return_route_stats=True,
        return_aggregator_stats=True,
        return_trajectory_prediction_stats=True,
    )

    assert output.logits.shape == (2, 4, 16)
    assert output.base_logits is not None
    assert torch.allclose(output.logits, output.base_logits, atol=1e-6)
    assert output.loss_components is not None
    assert abs(output.loss_components["ce_delta_vs_base"]) <= 1e-6
    assert output.trajectory_prediction_stats is not None
    assert output.trajectory_prediction_stats["fusion_improvement_mean"] == 0.0
    assert output.trajectory_prediction_stats["fusion_beats_base_rate"] == 0.0
    assert "fusion_regret_vs_oracle_mean" in output.trajectory_prediction_stats
    assert "alt_advantage_mean" in output.trajectory_prediction_stats
    assert "aggregator_alt_mass_alt_advantage_corr" in output.trajectory_prediction_stats
    assert output.aggregator_stats is not None
    assert output.aggregator_stats["residual_norm_ratio"] == 0.0
    noise_layers = [
        layer_idx
        for layer_idx, stats in output.route_stats.items()
        if stats["noise_applied"]
    ]
    assert noise_layers == [model.router_noise.target_layer_idx]


def test_loss_can_include_residual_and_trajectory_aux_terms() -> None:
    torch.manual_seed(13)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    base_logits = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    residual_l2 = torch.tensor(2.0, requires_grad=True)
    trajectory_aux = torch.tensor(3.0, requires_grad=True)
    loss, components = TrajectoryEnsembleForCausalLM._compute_loss(
        None,
        logits=logits,
        labels=labels,
        base_logits=base_logits,
        kl_beta=0.0,
        kl_direction="base_to_fused",
        residual_l2=residual_l2,
        residual_l2_weight=0.5,
        trajectory_oracle_aux_loss=trajectory_aux,
        trajectory_oracle_aux_weight=0.25,
        trajectory_oracle_aux_components={"trajectory_oracle_aux_ce": 3.0},
        aggregator_oracle_align_loss=torch.tensor(4.0, requires_grad=True),
        aggregator_oracle_align_weight=0.125,
        aggregator_oracle_align_components={"aggregator_oracle_align_ce": 4.0},
    )
    loss.backward()
    assert components["residual_l2"] == 2.0
    assert components["residual_l2_weight"] == 0.5
    assert components["trajectory_oracle_aux_ce"] == 3.0
    assert components["trajectory_oracle_aux_weight"] == 0.25
    assert components["aggregator_oracle_align_ce"] == 4.0
    assert components["aggregator_oracle_align_weight"] == 0.125
    assert residual_l2.grad is not None
    assert trajectory_aux.grad is not None
    assert logits.grad is not None


def test_selective_kl_weight_relaxes_preservation_where_alternatives_win() -> None:
    import torch.nn.functional as F

    torch.manual_seed(23)
    logits = torch.randn(1, 4, 7, requires_grad=True)
    base_logits = torch.randn(1, 4, 7)
    labels = torch.randint(0, 7, (1, 4))
    weight = torch.tensor([[1.0, 0.5, 0.0]])

    _, uniform_components = TrajectoryEnsembleForCausalLM._compute_loss(
        None,
        logits=logits,
        labels=labels,
        base_logits=base_logits,
        kl_beta=1.0,
        kl_direction="base_to_fused",
    )
    loss, components = TrajectoryEnsembleForCausalLM._compute_loss(
        None,
        logits=logits,
        labels=labels,
        base_logits=base_logits,
        kl_beta=1.0,
        kl_direction="base_to_fused",
        kl_token_weight=weight,
    )
    loss.backward()
    assert logits.grad is not None

    shift_logits = logits.detach()[..., :-1, :]
    shift_base = base_logits[..., :-1, :]
    kl_per_token = F.kl_div(
        F.log_softmax(shift_logits.float(), dim=-1),
        F.softmax(shift_base.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected_kl = (kl_per_token * weight).sum() / weight.sum()
    assert abs(components["kl"] - float(expected_kl)) < 1e-5
    assert components["kl"] != uniform_components["kl"]
    assert abs(components["kl_token_weight_mean"] - 0.5) < 1e-6


def test_forward_selective_kl_is_finite_and_bounded_at_identity_init() -> None:
    torch.manual_seed(24)
    base = TinyCausalLM()
    model = TrajectoryEnsembleForCausalLM(
        base_model=base,
        num_trajectories=3,
        agg_dim=4,
        noise_scale=0.4,
        top_k=2,
    )
    input_ids = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        kl_beta=1.0,
        kl_advantage_tau=0.5,
    )
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.loss_components is not None
    assert output.loss_components["kl_advantage_tau"] == 0.5
    assert 0.0 <= output.loss_components["kl_token_weight_mean"] <= 1.0
    # Identity init: fused logits equal base logits, so the weighted KL is 0.
    assert abs(output.loss_components["kl"]) <= 1e-6


def test_freeze_seed_noise_trains_aggregator_only_but_keeps_noise_active() -> None:
    torch.manual_seed(25)
    base = TinyCausalLM()
    model = TrajectoryEnsembleForCausalLM(
        base_model=base,
        num_trajectories=3,
        agg_dim=4,
        noise_scale=0.4,
        top_k=2,
    )
    freeze_seed_noise(model)
    trainable = [param for param in model.trainable_parameters() if param.requires_grad]
    aggregator_param_ids = {id(param) for param in model.aggregator.parameters()}
    assert trainable
    assert all(id(param) in aggregator_param_ids for param in trainable)
    assert len(trainable) == len(list(model.aggregator.parameters()))
    assert not model.router_noise.disable_noise
    assert float(model.router_noise.effective_noise().abs().sum()) > 0.0


def test_token_routing_trace_capture_shapes_and_alpha() -> None:
    torch.manual_seed(26)
    base = TinyCausalLM()
    model = TrajectoryEnsembleForCausalLM(
        base_model=base,
        num_trajectories=3,
        agg_dim=4,
        noise_scale=0.4,
        top_k=2,
    )
    input_ids = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_token_routing=True,
        return_pre_norm_hidden=True,
        return_aggregator_alpha=True,
    )
    assert output.token_routing is not None
    assert sorted(output.token_routing) == [1, 2]
    for payload in output.token_routing.values():
        assert payload["topk_idx"].shape == (2, 3, 4, 2)
        assert payload["topk_weight"].shape == (2, 3, 4, 2)
        assert int(payload["topk_idx"].max()) < 4
    assert output.pre_norm_by_traj is not None
    assert output.pre_norm_by_traj.shape == (2, 3, 4, 8)
    assert output.aggregator_alpha is not None
    assert output.aggregator_alpha.shape == (2, 3, 4)  # 2 alts + null
    assert torch.allclose(output.aggregator_alpha.sum(dim=1), torch.ones(2, 4), atol=1e-5)
    # Trace flags must not leak into later plain forwards.
    plain = model(input_ids=input_ids, attention_mask=attention_mask)
    assert plain.token_routing is None
    assert not model.router_noise.record_token_routing


def test_decontamination_ngram_filter_catches_eval_overlap() -> None:
    eval_question = "Natalia sold clips to 48 of her friends in April and then half as many in May"
    index = _DECONTAM.text_ngrams(eval_question, n=8)
    leaked = "Q: Natalia sold clips to 48 of her friends in April and then half as many in May. A: 72"
    clean = "The weather in April was unusually warm and many friends visited the park together."
    assert _DECONTAM.is_contaminated(leaked, index, n=8)
    assert not _DECONTAM.is_contaminated(clean, index, n=8)
    short_index = _DECONTAM.text_ngrams("what is two plus two", n=8)
    assert _DECONTAM.is_contaminated("what is two plus two", short_index, n=8)


def test_junk_filter_and_hard_prefix_selection() -> None:
    spam = (
        "Hi! On site stv24.info you can find Aryana with the service Role playing "
        "for date. Call Ximena or send a SMS and enjoy our dating service today."
    )
    stuffing = "battery powered light fixtures battery operated light fixture cordless lamp " * 4
    clean = (
        "The proof follows because the sequence is bounded and monotone, so it "
        "converges to a limit that we can then identify with the supremum of the set."
    )
    assert _DATA_QUALITY.looks_like_junk(spam)
    assert _DATA_QUALITY.looks_like_junk(stuffing)
    assert not _DATA_QUALITY.looks_like_junk(clean)

    def row(name, ce, dis, div):
        return {
            "text": clean + f" Case {name} considers what happens when the bound is not tight.",
            "mining": {
                "score": ce,
                "base_ce": ce,
                "gold_nll_std": dis,
                "route_divergence": div,
                "router_entropy": 1.0,
            },
        }

    scored = [
        row("A", 3.0, 0.01, 0.00),  # hard only because of raw CE
        row("B", 2.0, 0.50, 0.50),  # trajectories disagree -> should win
        row("C", 1.0, 0.05, 0.05),
        {"text": spam, "mining": {"score": 9.0, "base_ce": 9.0, "gold_nll_std": 0.0,
                                  "route_divergence": 0.0, "router_entropy": 9.0}},
    ]
    args = SimpleNamespace(
        keep_fraction=1.0,
        base_ce_percentile_cap=100.0,
        weight_base_ce=1.0,
        weight_nll_std=2.0,
        weight_route_div=1.0,
        weight_entropy=0.25,
    )
    selected = _MINER.select_hard(scored, args)
    assert len(selected) == 3  # spam row dropped
    assert selected[0]["mining"]["gold_nll_std"] == 0.50  # disagreement outranks raw CE
    assert all("score_raw" in row["mining"] for row in selected)


def test_evaluate_answer_extraction_and_code_truncation() -> None:
    assert _EVALUATE.extract_last_number("the answer is #### 1,234.5 ok") == "1234.5"
    assert _EVALUATE.extract_last_number("no digits here") is None
    assert _EVALUATE.numbers_match("42", "42.0")
    assert not _EVALUATE.numbers_match("41", "42")
    body = "    return x + 1\n\ndef next_function():\n    pass"
    assert _EVALUATE.truncate_code(body) == "    return x + 1\n"


def test_encode_batch_masks_prompt_but_keeps_completion_targets() -> None:
    tokenizer = ToyTokenizer()
    batch = encode_batch(
        examples=[TrainingExample(prompt="question", target=" answer")],
        tokenizer=tokenizer,
        max_length=16,
        train_on_prompt=False,
        append_eos_to_target=True,
        min_target_tokens=1,
        device=torch.device("cpu"),
    )
    assert batch is not None
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    assert input_ids.shape == labels.shape == attention_mask.shape
    active_labels = labels[0, attention_mask[0].bool()]
    assert active_labels[0].item() == -100
    assert active_labels[1].item() == -100
    assert active_labels[2:].ne(-100).all()
    assert labels[:, 1:].ne(-100).sum().item() >= 1


def test_encode_batch_can_train_on_full_text_or_prompt() -> None:
    tokenizer = ToyTokenizer()
    batch = encode_batch(
        examples=[TrainingExample(prompt="question", target=" answer")],
        tokenizer=tokenizer,
        max_length=16,
        train_on_prompt=True,
        append_eos_to_target=False,
        min_target_tokens=1,
        device=torch.device("cpu"),
    )
    assert batch is not None
    active = batch["attention_mask"][0].bool()
    assert batch["labels"][0, active].ne(-100).all()


def test_encode_batch_supports_last_assistant_message_as_target() -> None:
    tokenizer = ToyTokenizer()
    batch = encode_batch(
        examples=[
            TrainingExample(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            )
        ],
        tokenizer=tokenizer,
        max_length=16,
        train_on_prompt=False,
        append_eos_to_target=True,
        min_target_tokens=1,
        device=torch.device("cpu"),
    )
    assert batch is not None
    active_labels = batch["labels"][0, batch["attention_mask"][0].bool()]
    assert active_labels[:-1].eq(-100).any()
    assert active_labels[-1].item() != -100


def test_grad_summary_reports_adapter_group_gradient_flow() -> None:
    module = nn.Linear(3, 2)
    empty = grad_summary(module)
    assert empty["tensors"] == 2
    assert empty["tensors_with_grad"] == 0
    assert empty["l2_norm"] == 0.0

    x = torch.randn(4, 3)
    module(x).float().pow(2).mean().backward()
    filled = grad_summary(module)
    assert filled["tensors"] == 2
    assert filled["tensors_with_grad"] == 2
    assert filled["l2_norm"] > 0.0


if __name__ == "__main__":
    test_aggregator_identity_and_initial_gradient()
    test_aggregator_can_disable_null_candidate()
    test_aggregator_delta_value_ignores_identical_alt_hidden()
    test_orthogonal_seed_init_is_centered_orthogonal_and_scaled()
    test_relative_keys_center_scores_and_keep_identity_init()
    test_first_router_noise_is_centered_and_base_is_zero()
    test_disable_seed_noise_is_true_ablation()
    test_context_seed_gate_is_identity_initialized_and_bounded()
    test_context_seed_gate_gets_gradient_through_moe_output()
    test_first_router_noise_gets_gradient_through_moe_output()
    test_first_router_st_topk_forward_matches_hard_eval_but_keeps_dense_gradient()
    test_first_router_soft_all_training_surrogate_switches_to_hard_eval()
    test_sorted_moe_infer_matches_naive_weighted_sum()
    test_route_stats_ignore_padding_tokens()
    test_context_gate_regularization_is_zero_init_then_positive()
    test_path_prefix_remembers_earlier_divergence()
    test_trajectory_stats_are_masked_and_finite()
    test_symmetric_kl_has_gradient_to_fused_logits()
    test_loss_handles_all_ignored_labels_without_nan()
    test_loss_handles_single_token_sequence_with_kl()
    test_residual_l2_ratio_is_masked()
    test_prediction_stats_capture_oracle_and_top1_diversity()
    test_aggregator_alignment_stats_compare_alpha_to_oracle_nll()
    test_fusion_improvement_stats_connect_oracle_advantage_to_fused_gain()
    test_forward_returns_fusion_conversion_stats_with_identity_init()
    test_loss_can_include_residual_and_trajectory_aux_terms()
    test_selective_kl_weight_relaxes_preservation_where_alternatives_win()
    test_forward_selective_kl_is_finite_and_bounded_at_identity_init()
    test_freeze_seed_noise_trains_aggregator_only_but_keeps_noise_active()
    test_token_routing_trace_capture_shapes_and_alpha()
    test_decontamination_ngram_filter_catches_eval_overlap()
    test_junk_filter_and_hard_prefix_selection()
    test_evaluate_answer_extraction_and_code_truncation()
    test_encode_batch_masks_prompt_but_keeps_completion_targets()
    test_encode_batch_can_train_on_full_text_or_prompt()
    test_encode_batch_supports_last_assistant_message_as_target()
    test_grad_summary_reports_adapter_group_gradient_flow()
    print("[tests] math components ok")
