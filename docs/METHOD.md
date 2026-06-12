# Seeded Latent Trajectory Aggregation

## Core Claim

The model creates multiple latent reasoning trajectories inside a single
DeepSeek-MoE backbone by perturbing only the first MoE router. The first route
is the unmodified base route. Other routes receive small trainable expert-logit
seed biases. After that first routing event, all layers use the original frozen
routers and experts.

This makes the method a seeded internal trajectory ensemble:

```text
first router seed -> different hidden states -> natural downstream routing
divergence -> final latent aggregation -> original LM head
```

## Mathematical Invariants

- Base trajectory is always trajectory 0 and receives exactly zero noise.
- Seed noise is centered per trajectory, so it expresses relative expert
  preference rather than a useless common logit shift.
- Seed rows are initialized pairwise-orthogonal inside the zero-sum
  expert-logit subspace by default (`seed_init_mode=orthogonal`), with each row
  rescaled to the same per-element std as the Gaussian init. Candidate latent
  views therefore start maximally separated as an inductive bias, without any
  diversity loss pressure. `gaussian` remains available as an ablation.
- Each non-base trajectory has a positive bounded trainable seed scale.
- By default, the seed scale also has an identity-initialized bounded context
  gate. The trajectory seed direction remains fixed, while prompt context can
  adjust how strongly that direction is applied:

```text
epsilon_p(x) = s_p * m_p(x) * center(noise_p)
m_p(x) = 1 + delta tanh(Wc LN(mean_valid hidden_base))
```

- Noise is injected before softmax and top-k, matching the canonical location
  of noisy sparse MoE gating.
- DeepSeek backbone, router weights, experts, shared experts, final norm, and
  LM head are frozen.
- The aggregator operates before DeepSeek's original final norm. No extra final
  normalization is introduced.
- The aggregator is identity-initialized: `Wo=0`, so the initial fused hidden is
  exactly the base pre-norm hidden.
- The residual scale is bounded as `lambda = lambda_max * tanh(raw_lambda)`,
  keeping the fused hidden close to the frozen LM head's expected space.

## Aggregator

For final pre-norm hidden states `h_p,t`, with `p=0` as the base route:

```text
q_tok,t = Wq LN(h_0,t)
q_ctx   = Wg mean_valid_t LN(h_0,t)
q_t     = q_tok,t + q_ctx
k_p,t   = Wk LN(h_p,t), p > 0
v_p,t   = Wv (LN(h_p,t) - LN(h_0,t)), p > 0
score_p = q_t^T k_p,t / sqrt(d), p > 0
score_null = q_t^T k_null / sqrt(d)
alpha = softmax([score_1 ... score_P score_null])
delta_t = Wo sum_p alpha_p v_p,t
h_fused = h_0,t + lambda * delta_t
logits  = LMHead(FinalNorm(h_fused))
```

The base route is not included as a value candidate. It is the residual anchor.
The noisy routes act as alternative latent views that the aggregator may read.
By default the value is base-relative (`aggregator_value_mode=delta`), so an
alternative trajectory contributes only its contrastive difference from the base
route. `absolute` mode is exposed as an ablation.
The null candidate has a learned key but zero value, so the aggregator can
abstain from using noisy trajectories at a token position without rewriting the
base hidden state.
The query also receives a masked-mean base-route context term, so trajectory
selection can depend on the whole prompt/problem rather than only a local token
vector.

With `aggregator_relative_keys=True`, trajectory keys are centered across the
candidate axis before scoring:

```text
k_p,t <- k_p,t - mean_p' k_p',t
```

so the judge scores each alternative relative to the other candidates instead
of absolutely; a key direction shared by every noisy trajectory carries no
selection signal. The null key is not centered, so abstention keeps its own
reference point. This comparative-judge variant is off by default and exposed
as an ablation.

## Loss

The default training loss preserves the frozen model behavior while allowing the
adapter to improve next-token prediction:

```text
L = CE(p_fused, labels) + beta KL(p_base || p_fused)
```

`beta` is scheduled from `1.0` to `0.1` in `scripts/train_adapter.py`.
`base_to_fused` is the default KL direction; `symmetric` is available for a
stricter drift penalty.

An optional advantage-weighted KL is implemented (`--kl_advantage_tau`,
default `0.0` = uniform):

```text
adv_t = NLL_base,t - min_p>0 NLL_p,t
w_t   = stopgrad sigmoid(-adv_t / tau_kl)
L_KL  = beta * sum_t w_t KL_t(p_base || p_fused) / sum_t w_t
```

Uniform KL fights fusion exactly on the tokens where alternatives carry
demonstrably better gold-token evidence. Advantage weighting keeps
preservation strong where the base route wins and relaxes it where the
alternatives win, which matches the selective-fusion reading of the method.
Like the oracle auxiliaries, it needs per-trajectory LM-head NLLs (computed
under `no_grad`), so it adds LM-head passes per training step and is a
label-aware training-time weighting only.
The training log also records `base_ce` and `ce_delta_vs_base`, so improvement
is judged against the frozen base trajectory rather than only by absolute CE.

The training script now supports instruction-style target masking. For paired
examples such as `prompt+completion`, `question+answer`, `instruction+output`,
or chat `messages` whose last turn is assistant, prompt tokens are masked with
`-100` and only target tokens contribute next-token CE by default:

```text
input_ids = [prompt_ids, target_ids]
labels    = [-100 ... -100, target_ids]
```

The prompt and target are tokenized separately before concatenation. This avoids
BPE boundary merges that can accidentally mask the first target token.
`--train_on_prompt` is available as an explicit ablation.

Two adapter-stability terms are exposed in the model forward/training CLI:

```text
R_hidden = mean_valid ||h_fused - h_base||_2^2 / ||h_base||_2^2
L = CE + beta KL + rho R_hidden
```

`R_hidden` is a hidden-manifold safety rail for the frozen final norm and LM
head. It complements output-space KL: KL constrains next-token distributions,
while `R_hidden` constrains the representation actually given to the frozen
unembedding path.

An optional trajectory-level auxiliary loss is also implemented:

```text
NLL_p,t = -log p_p(y_t | x_<t), p > 0
L_oracle = mean_valid [ -tau log sum_p exp(-NLL_p,t / tau) + tau log(P) ]
```

This is a smooth multiple-choice CE over the noisy trajectories only. It gives
the first-router seed direct task credit even before the aggregator has learned
to read every trajectory well. It is disabled by default
(`--trajectory_oracle_aux_weight 0.0`) because it adds LM-head passes and can
fight trajectory diversity if overweighted. The main paper setting should treat
it as a training-stability ablation unless experiments show it is necessary.

The aggregator can also be supervised directly with an optional oracle-alpha
alignment loss:

```text
z_p,t    = -NLL_p,t / tau, p > 0
z_null,t = -NLL_base,t / tau
q*_t     = stopgrad softmax([z_1,t ... z_P,t z_null,t])
L_align  = mean_valid CE(q*_t, alpha_t)
```

If the null candidate is enabled, the frozen base route acts as the oracle
target for "do not use noisy trajectories" cases. This loss trains the
aggregator's query/key/null-key judge even when the residual output projection
is still near its identity initialization. It is also disabled by default
(`--aggregator_oracle_align_weight 0.0`) because it is a label-aware auxiliary
teacher; it should be used as an ablation or stabilization option, not as a
substitute for reporting fused CE.

The training log records:

```text
loss.residual_l2
loss.trajectory_oracle_aux_ce
loss.trajectory_oracle_candidate_ce
trajectory_prediction.ce_by_traj
trajectory_prediction.oracle_ce
trajectory_prediction.top1_match_with_base
trajectory_prediction.aggregator_alpha_on_best_alt_mean
trajectory_prediction.aggregator_alt_oracle_regret_mean
trajectory_prediction.aggregator_alt_mass_mean_on_alt_better
trajectory_prediction.aggregator_null_alpha_mean_on_base_better
trajectory_prediction.fusion_improvement_mean
trajectory_prediction.fusion_regret_vs_oracle_mean
trajectory_prediction.alt_advantage_mean
trajectory_prediction.fusion_improvement_alt_advantage_corr
trajectory_prediction.aggregator_alt_mass_alt_advantage_corr
```

This separates three questions: whether routes differ, whether hidden states
differ, whether the alternative hidden states contain complementary next-token
evidence, and whether the aggregator actually assigns mass to the useful
trajectory or null candidate.

The prediction logger also measures whether useful alternatives become actual
fused-output gains:

```text
alt_advantage_t = NLL_base,t - min_p>0 NLL_p,t
fusion_gain_t   = NLL_base,t - NLL_fused,t
fusion_regret_t = NLL_fused,t - min_p>=0 NLL_p,t
```

If `alt_advantage_t > 0` but `fusion_gain_t <= 0`, then useful latent evidence
exists but the residual fusion path failed to convert it into a better
next-token distribution. The log also records correlations between
`alt_advantage_t`, aggregator alt mass, and fused improvement. This follows the
recent ensemble-confidence lesson that fusion should be selective and
confidence-aware rather than applied blindly at every token.

A small seed diversity regularizer is added in the training script:

```text
L_seed = mean offdiag(cos(center(noise_i), center(noise_j))^2)
```

This discourages seed collapse without directly forcing any downstream expert
path. Route divergence itself is logged as a metric, not used as a default loss.
Final hidden-state divergence is also logged because routing diversity is only a
proxy; the actual claim concerns distinct vector representations.
Layerwise top-k exact-match metrics are logged alongside overlap metrics because
expert paths, not just one-layer expert usage, are the relevant object.
The training script also adds a tiny context-gate L2 prior by default, so the
context seed gate must earn movement through CE rather than drifting freely.

During training, the first MoE router defaults to a straight-through top-k
surrogate:

```text
w_train = stopgrad(w_hard) + w_soft - stopgrad(w_soft)
```

This preserves the hard sparse top-k forward trajectory while giving all
first-layer experts a dense softmax gradient. `soft_all` remains available as a
stronger relaxation, but `st_topk` is the default.

The first-layer training forward value is sparse, but the surrogate evaluates
all first-layer experts so the seed router can receive dense counterfactual
credit assignment. Later layers keep the original sparse top-k computation.

`Wo=0` also means the first task-loss step updates only the output projection
of the aggregator; task gradients reach q/k/v and seed noise after `Wo` becomes
nonzero. This is the cost of exact base-logit identity at initialization.

## Implementation Notes

- DeepSeek's original eval MoE infer path is decorated with `no_grad`. The
  wrapper replaces `DeepseekMoE.forward` with a differentiable inference path so
  gradients can reach selected gate weights and first-router seed noise.
- The replacement MoE infer uses expert-id sorting plus `index_add_`, preserving
  gradients while avoiding a full expert-by-expert mask scan.
- Routing and path metrics are attention-mask aware, so padding tokens do not
  contaminate divergence estimates.
- Aggregator alpha and residual statistics are also attention-mask aware.
- Aggregator logs include null-candidate alpha and total noisy-trajectory alpha
  mass when `include_null_aggregation_candidate=True`.
- Aggregator values default to base-relative deltas. This prevents the adapter
  from relearning common base information through the alternative trajectories.
- Router boundary logging includes logit margins as well as probability margins,
  because seed noise is applied in logit space.
- First-router logging includes `seed_scale_by_traj`, which is the realized
  per-trajectory seed scale after the context gate.
- Routing logs include expert-distribution JSD against the base trajectory. This
  helps distinguish useful path diversity from simple expert-distribution drift
  or collapse.
- Trajectory prediction logs include per-trajectory CE and oracle CE. This
  checks whether latent diversity is predictive diversity rather than only
  geometric or routing diversity.
- Aggregator-oracle alignment logs compare token-level alpha against
  trajectory NLLs. This separates seed failure from latent-judge failure.
- Fusion-improvement logs compare fused gold-token NLL against base and oracle
  trajectory NLLs. This separates latent-judge failure from residual/value
  conversion failure.
- Path metrics track cumulative prefix exact-match with the base route. A later
  layer can match the base layerwise while still belonging to a different
  overall expert path.
- `--disable_seed_noise` is available for the critical aggregator-only
  ablation. In that mode, all trajectories follow the base route and the seed
  noise parameters are excluded from optimization.
- `--freeze_seed_noise` is the sharper frozen-random-seed control: the entire
  `SeedRouterNoise` module (seed vectors, scales, context gate) keeps its
  initialization and stays active in the forward pass, but only the aggregator
  is optimized. This separates "the perturbation must be learned" from "the
  perturbation must exist".
- Supported trajectory counts are `N=3` and `N=5`.
- See `docs/METHOD_AUDIT.md` for the explicit mathematical failure modes and
  the refinements used to avoid overclaiming.

## Reference Anchors

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer", 2017:
  https://arxiv.org/abs/1701.06538
- Fedus et al., "Switch Transformers", 2021/2022:
  https://arxiv.org/abs/2101.03961
- Zhou et al., "Mixture-of-Experts with Expert Choice Routing", 2022:
  https://arxiv.org/abs/2202.09368
- Gu et al., "Path-Constrained Mixture-of-Experts", 2026:
  https://arxiv.org/abs/2603.18297
- Wu et al., "ReFT: Representation Finetuning for Language Models", 2024:
  https://arxiv.org/abs/2404.03592
- Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and
  Self-Speculative Decoding", ACL 2024:
  https://aclanthology.org/2024.acl-long.681/
- Fein-Ashley et al., "Mixture of Thoughts: Learning to Aggregate What Experts
  Think, Not Just What They Say", 2025:
  https://arxiv.org/abs/2509.21164
- Li et al., "DLLG: Dynamic Logit-Level Gating of LLM Experts", 2026:
  https://arxiv.org/abs/2606.04378
- Guo, "LENS: Learning Ensemble Confidence from Neural States for Multi-LLM
  Answer Integration", 2025:
  https://arxiv.org/html/2507.23167v1
- Yun et al., "When to Ensemble: Identifying Token-Level Points for Stable and
  Fast LLM Ensembling", ICLR 2026:
  https://openreview.net/forum?id=kfPkF2ACDM
