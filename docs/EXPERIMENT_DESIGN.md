# Experiment Design For Seeded Latent Trajectory Aggregation

This document defines what would count as evidence for or against the method.
It is intentionally stricter than an implementation checklist.

## Core Hypothesis

The defensible hypothesis is:

```text
A small trainable first-router seed can induce multiple same-latent-space MoE
expert paths whose final hidden states contain complementary next-token
evidence. A bounded base-anchored aggregator can exploit that evidence without
leaving the frozen LM head's expected representation space.
```

This is not the same as claiming independent-model ensemble equivalence.

## Required Evidence Chain

The method needs all four links below. If any link fails, the claim should be
weakened.

1. Seed controllability:

```text
first-layer exact top-k match with base decreases for non-base trajectories
```

2. Path persistence:

```text
prefix exact-match with base remains below 1.0 through later MoE layers
```

3. Representation divergence:

```text
final base-alt cosine is measurably below 1.0
and base-alt L2 ratio is nonzero on valid tokens
```

4. Useful fusion:

```text
CE(fused) improves over CE(base) or CE(base-only adapter ablation)
while KL and residual/base norm ratio remain bounded
```

5. Predictive complementarity:

```text
trajectory oracle CE is below base CE on at least some target tokens
and fused CE moves toward that oracle without excessive KL/residual drift
```

6. Judge alignment:

```text
aggregator alpha puts more mass on low-NLL alternative trajectories
and on the null candidate when the base trajectory is best
```

7. Fusion conversion:

```text
fused gold-token NLL improves most when alternative trajectories beat the base
and fused regret versus the per-token trajectory oracle stays bounded
```

Routing divergence without representation divergence is not enough. Hidden
divergence without CE improvement is not enough. CE improvement with excessive
KL or residual norm is likely an unsafe hidden-space rewrite rather than a
clean latent trajectory ensemble. If trajectory oracle CE improves but fused CE
does not, the seed is producing useful alternatives and the aggregator is the
weak link. If trajectory oracle CE does not improve, the first-router seed is
not creating useful predictive alternatives for that dataset. If trajectory
oracle CE improves but alpha ignores the best alternative, the latent judge is
the weak link rather than the seed. If alpha aligns with useful alternatives
but fused NLL does not improve, the failure is downstream of judging: value
projection, residual scale, final-norm/LM-head compatibility, or an overly
conservative KL/residual constraint.

## Primary Metrics

- `route_stats[layer].topk_overlap_with_base`
- `route_stats[layer].topk_exact_match_with_base`
- `route_stats[layer].topk_margin_by_traj`
- `route_stats[layer].topk_logit_margin_by_traj`
- `route_stats[layer].seed_scale_by_traj`
- `route_stats[layer].expert_jsd_with_base`
- `path_stats[layer].prefix_exact_match_with_base`
- `path_stats[layer].new_divergence_rate_from_previous`
- `trajectory.base_alt_cosine_mean`
- `trajectory.base_alt_l2_ratio_mean`
- `trajectory.alt_alt_cosine_mean`
- `aggregator.alpha_mean`
- `aggregator.alpha_std`
- `aggregator.alpha_entropy_mean`
- `aggregator.alpha_max_mean`
- `aggregator.null_alpha_mean`
- `aggregator.alt_alpha_mass_mean`
- `aggregator.residual_norm_ratio`
- `loss.ce`
- `loss.base_ce`
- `loss.ce_delta_vs_base`
- `loss.kl`
- `loss.kl_token_weight_mean` (when `--kl_advantage_tau > 0`)
- `loss.residual_l2`
- `loss.trajectory_oracle_aux_ce`
- `loss.trajectory_oracle_candidate_ce`
- `loss.aggregator_oracle_align_ce`
- `loss.total_with_reg`
- `loss.context_gate_l2`
- `loss.context_gate_norm`
- `grad.router_noise.l2_norm`
- `grad.router_noise.tensors_with_grad`
- `grad.aggregator.l2_norm`
- `grad.aggregator.tensors_with_grad`
- `trajectory_prediction.ce_by_traj`
- `trajectory_prediction.ce_delta_vs_base_by_traj`
- `trajectory_prediction.oracle_ce`
- `trajectory_prediction.oracle_ce_delta_vs_base`
- `trajectory_prediction.gold_nll_std_mean`
- `trajectory_prediction.gold_nll_range_mean`
- `trajectory_prediction.top1_match_with_base`
- `trajectory_prediction.aggregator_alpha_on_best_alt_mean`
- `trajectory_prediction.aggregator_alt_oracle_regret_mean`
- `trajectory_prediction.aggregator_alpha_best_alt_top1_match`
- `trajectory_prediction.aggregator_alpha_oracle_corr_mean`
- `trajectory_prediction.aggregator_alt_better_rate`
- `trajectory_prediction.aggregator_alt_mass_mean_on_alt_better`
- `trajectory_prediction.aggregator_alt_mass_mean_on_base_better`
- `trajectory_prediction.aggregator_null_alpha_mean_on_alt_better`
- `trajectory_prediction.aggregator_null_alpha_mean_on_base_better`
- `trajectory_prediction.fusion_improvement_mean`
- `trajectory_prediction.fusion_beats_base_rate`
- `trajectory_prediction.fusion_regret_vs_oracle_mean`
- `trajectory_prediction.fusion_regret_vs_alt_oracle_mean`
- `trajectory_prediction.alt_advantage_mean`
- `trajectory_prediction.alt_advantage_positive_rate`
- `trajectory_prediction.fusion_improvement_on_alt_better`
- `trajectory_prediction.fusion_improvement_on_base_better`
- `trajectory_prediction.fusion_improvement_alt_advantage_corr`
- `trajectory_prediction.aggregator_alt_mass_alt_advantage_corr`

All routing and path metrics must be attention-mask aware.

## Minimal Ablations

The first experiments should compare:

- Base frozen DeepSeek logits.
- Aggregator only, with all seed noise disabled:
  `scripts/train_adapter.py --disable_seed_noise`.
- Frozen random seeds, aggregator-only optimization:
  `scripts/train_adapter.py --freeze_seed_noise`. Trained seeds must beat this
  control, or seed learning is not contributing beyond random perturbation.
- Seeded trajectories with `N=3`.
- Seeded trajectories with `N=5`.
- `train_router_mode=hard` versus `st_topk`.
- `--context_seed_gate` versus `--no-context_seed_gate`.
- `--seed_init_mode orthogonal` versus `--seed_init_mode gaussian`.
- `--include_null_aggregation_candidate` versus `--no-include_null_aggregation_candidate`.
- `--aggregator_value_mode delta` versus `--aggregator_value_mode absolute`.
- `--aggregator_relative_keys` versus `--no-aggregator_relative_keys`.
- `kl_direction=base_to_fused` versus `symmetric`.
- `--kl_advantage_tau 0.0` versus a small value such as `0.5`.
- `--trajectory_oracle_aux_weight 0.0` versus a small value such as `0.02`.
- `--aggregator_oracle_align_weight 0.0` versus a small value such as `0.02`.
- target-only labels versus `--train_on_prompt` for paired instruction data.

The most important negative control is the aggregator-only setting. If it
matches the full method, the claimed benefit is not from seeded trajectory
divergence. In that case the method should be reframed as a small residual
adapter, not as a latent trajectory ensemble. The second-most important
control is `--freeze_seed_noise`: if random fixed seeds match trained seeds,
the method is noise ensembling with a learned judge, not learned latent
perspectives.

## Target Evaluation Data

The research claim is that aggregated latent views help on challenging,
heterogeneous inputs — the latent analogue of querying several different
models. A single-domain eval cannot test this: one fixed seed direction could
simply overfit that domain. The target suite should therefore be multi-domain
and difficulty-skewed, for example:

- MMLU-Pro: multi-domain, reasoning-heavy multiple choice.
- BBH (BIG-Bench Hard): heterogeneous reasoning task collection.
- GPQA-Diamond: graduate-level science questions.
- MATH-500: competition mathematics.
- MuSR: multi-step soft reasoning over long narratives.

Reporting rules for this suite:

- Report per-domain/per-task deltas versus the frozen base, not only suite
  averages. The heterogeneous-ensemble framing makes a specific prediction:
  `alt_alpha_mass`, realized `seed_scale_by_traj`, and fused NLL gains should
  concentrate on the domains and tokens where the base model is weakest. A
  uniform gain across domains is more consistent with a generic residual
  adapter than with conditional latent views.
- Run the aggregator-only (`--disable_seed_noise`) and frozen-random-seed
  (`--freeze_seed_noise`) controls on the same suite with the same budget; the
  claim survives only if trained seeds beat both.
- Training data for the adapter should also be domain-mixed; otherwise the
  context seed gate cannot learn domain-conditional seed strength.

## Failure Criteria

The method is likely unsupported if any of these hold after a reasonable
adapter training schedule:

- First-layer exact-match with base remains near 1.0 and top-k logit margins
  are larger than the learned seed scale.
- The context seed gate saturates at its bounds but path divergence and CE do
  not improve.
- Prefix exact-match drops at the first MoE layer but final base-alt cosine
  stays near 1.0.
- Aggregator alpha collapses to a constant distribution and residual/base norm
  stays near zero while CE does not improve.
- Aggregator output-projection gradients are nonzero but router-noise gradients
  remain zero after the initial identity-preserving steps; this means the seed
  is not receiving task credit through either fused CE or auxiliary trajectory
  losses.
- Null alpha dominates across all tokens and CE does not improve, which means
  the latent alternatives are being ignored.
- Expert JSD rises sharply while final hidden divergence and CE do not improve;
  this suggests route drift rather than useful path complementarity.
- Trajectory oracle CE is not below base CE, which means the alternative hidden
  states do not contain better gold-token evidence even before aggregation.
- Trajectory oracle CE is below base CE but fused CE does not improve; this
  points to aggregator capacity/training failure rather than seed failure.
- Alpha on the best alternative remains low when `aggregator_alt_better_rate`
  is high; this means useful trajectories exist but the latent judge does not
  select them.
- Null alpha stays high even when alternatives beat the base; this means the
  null candidate has become a collapse route.
- `alt_advantage_mean` is positive but `fusion_improvement_mean` is near zero
  or negative; this means complementary evidence exists but is not converted
  into better next-token probabilities.
- `fusion_regret_vs_oracle_mean` stays large even when alpha-oracle alignment
  is good; this points to value-stream or residual-projection failure rather
  than seed or judge failure.
- `aggregator_alt_mass_alt_advantage_corr` is near zero or negative; this
  means fusion is not selectively applied on tokens where alternatives have
  predictive advantage.
- CE improves only when residual/base norm becomes large or KL rises sharply.
- `N=5` adds compute but does not improve path diversity or CE over `N=3`.

These are not implementation failures; they are ways the scientific hypothesis
can be false for this backbone/task/data regime.

## Interpretation Rules

- Treat expert paths as the main routing object, not isolated layer choices.
- Treat final hidden divergence as the representation-level object, not route
  divergence alone.
- Treat base-logit KL and residual/base norm as safety rails for the frozen LM
  head manifold.
- Treat trajectory oracle CE as a diagnostic upper bound for the current
  aggregator, not as the reported fused-model metric.
- Treat oracle-alpha alignment as a judge diagnostic. It should explain whether
  fusion failed because alternatives were bad or because the aggregator ignored
  useful alternatives.
- Treat fused gold-token NLL improvement as the actual next-token result. The
  oracle and alpha metrics are explanatory; the method succeeds only when
  useful alternatives are converted into fused-logit gains under bounded
  KL/residual drift.
- Treat selective fusion as desirable. Always using noisy alternatives is not a
  success criterion; using them on tokens with positive alternative advantage
  is the intended behavior.
- Treat `st_topk` as a training estimator: its forward value matches hard top-k,
  but the first MoE layer pays dense expert compute during training.

## Reference Anchors

- Sparse MoE routing and noisy top-k gating:
  https://arxiv.org/abs/1701.06538
- Switch-style sparse routing stability:
  https://arxiv.org/abs/2101.03961
- Differentiable expert selection:
  https://arxiv.org/abs/2106.03760
- Dense routing gradients for sparse MoE:
  https://arxiv.org/abs/2504.12463
- Expert paths as a MoE design axis:
  https://arxiv.org/abs/2603.18297
- Representation-level fine-tuning on frozen LMs:
  https://arxiv.org/abs/2404.03592
- Shared LM-head auxiliary losses in early-exit LMs:
  https://aclanthology.org/2024.acl-long.681/
- Latent-level aggregation of model hidden states:
  https://arxiv.org/abs/2509.21164
- Token-level expert fusion/gating:
  https://arxiv.org/abs/2606.04378
- Hidden-state confidence for latent/model ensemble weighting:
  https://arxiv.org/html/2507.23167v1
- Token-level selective ensembling:
  https://openreview.net/forum?id=kfPkF2ACDM
- Hidden-state structural stability as a confidence signal:
  https://arxiv.org/html/2602.00977v1
- Load-balancing implementation risks for MoE specialization:
  https://aclanthology.org/2025.acl-long.249/
- Context-independence risk in MoE routing:
  https://arxiv.org/abs/2402.01739
- Open MoE routing specialization analyses:
  https://arxiv.org/abs/2409.02060
