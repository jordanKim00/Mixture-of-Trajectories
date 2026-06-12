# Mathematical Audit And Method Refinement

## 1. Overclaim Risk: This Is Not A Fully Independent Model Ensemble

Naive claim:

```text
Different MoE trajectories are equivalent to ensembling different specialist LLMs.
```

This is too strong. The trajectories share attention weights, expert weights,
normalization, embeddings, and LM head. They are conditionally different paths
inside one parameterized function, not independently trained predictors.

Defensible claim:

```text
Seeded MoE routing creates a same-latent-space conditional path ensemble.
It may recover some benefits of external ensembling by producing diverse
intermediate representations without leaving the frozen model's hidden space.
```

The method should be evaluated as an efficient internal path ensemble, not as a
mathematically identical substitute for independently trained LLM ensembles.

## 2. Hard Top-k Gradient Problem

Hard top-k routing is piecewise constant in the selected expert indices. The
selected weights receive gradient, and the softmax denominator gives some signal
to unselected logits, but the model does not directly observe the functional
effect of currently unselected experts.

This weakens the learning signal for trainable seed noise.

Refinement implemented:

```text
default training first router: st_topk surrogate
evaluation first router: original hard top-k
```

For the first MoE layer only, training can use a straight-through dense
backprop surrogate:

```text
w_hard_full = scatter_topk_weights(router_logits + seed_noise)
w_soft      = mass_hard_topk.detach() * softmax((router_logits + seed_noise) / tau)
w_train     = stopgrad(w_hard_full) + w_soft - stopgrad(w_soft)
```

The forward value is exactly the sparse hard top-k route. The backward gradient
is dense through `w_soft`, so every first-layer expert receives a functional
credit-assignment signal. `mass_hard_topk` preserves the routed-output scale of
DeepSeek's `norm_topk_prob=false` path.

The older `soft_all` mode remains available as a stronger relaxation, but
`st_topk` is the default because it avoids a forward/inference mismatch in the
latent trajectory definition.

## 3. Hidden-Space Manifold Risk

If aggregation freely rewrites the final hidden vector, the frozen LM head may
receive vectors outside its trained distribution.

Refinement implemented:

```text
h_fused = h_base + lambda * delta
lambda = lambda_max * tanh(raw_lambda)
```

The output projection is zero-initialized, so the initial model is exactly the
frozen base model. `lambda` is bounded, so the aggregator is structurally a
correction, not a replacement.

## 3.5 Seed Strength Must Be Bounded But Learnable

A fixed seed scale is mathematically brittle. If the first router's top-k
boundary margin is large, the seed may never change expert selections. If the
scale is too large, it can destroy the base route's hidden-space neighborhood.

Refinement implemented:

```text
epsilon_p(x) = s_p * m_p(x) * center(noise_p)
s_p = s_max * sigmoid(raw_s_p)
m_p(x) = 1 + delta tanh(Wc LN(mean_valid hidden_base))
```

Each non-base trajectory learns its own positive bounded base seed scale. The
context gate is zero-initialized, so `m_p(x)=1` at initialization. During
adapter training, it can increase or decrease the strength of each trajectory
seed by prompt context without changing the trajectory seed direction. The
method logs top-k logit boundary margins and realized `seed_scale_by_traj`, so
the seed strength can be interpreted against the router's actual selection
stability in the same units as the perturbation.
The training script also exposes a small context-gate L2 prior, which
discourages unearned saturation while still allowing task loss to use the gate.

## 4. Diversity Is Necessary But Not Sufficient

Different first-router seeds do not guarantee useful downstream disagreement.
They can collapse to similar routing paths, or produce diversity that hurts CE.

Refinement implemented:

- Seed vectors are centered per trajectory.
- A small cosine-collapse penalty is available for seed vectors.
- Expert-distribution JSD against the base trajectory is logged, so routing
  divergence can be separated from broad expert-distribution drift or collapse.
- Route divergence is logged layer-by-layer but is not forced as a default loss.

This keeps diversity as an inductive bias, not an objective that can fight task
likelihood.

## 5. KL Direction Matters

The preservation term can be interpreted in two ways:

```text
KL(p_base || p_fused): preserve base-supported tokens
symmetric KL: additionally penalize fused-only distribution drift
```

The implementation defaults to `base_to_fused` because it is the usual
distillation direction and is less restrictive. `symmetric` is exposed for
settings where hidden-space safety matters more than adapter freedom.

## 6. Recommended Final Method Statement

Use this framing:

```text
Seeded Latent Trajectory Aggregation turns one frozen MoE LLM into a
same-latent-space path ensemble. Trainable seed biases perturb only the first
MoE routing decision. The resulting hidden-state perturbation is then amplified
or damped by the model's own frozen attention and routing dynamics. A small
base-anchored residual judge reads alternative final hidden states and injects
a bounded correction before the original final norm and LM head.
```

## 7. Why First-Router-Only Noise Is Plausible But Falsifiable

The first MoE perturbation changes the routed FFN contribution:

```text
h_1^p = h_0 + Attn_0(h_0) + MoE_1(h_0; topk(router(h_0) + eps_p))
```

For later layers, no explicit noise is injected:

```text
h_l^p = F_l(h_{l-1}^p; theta_frozen)
```

Thus route diversity is not assumed to persist by construction. It is an
empirical property of the frozen model dynamics. The method logs:

- first-layer top-k overlap with the base route
- first-layer exact top-k set match with the base route
- top-k probability and logit boundary margins between the kth and k+1th experts
- later-layer top-k overlap with the base route
- later-layer exact top-k set match with the base route
- router entropy by trajectory
- expert-distribution JSD against the base trajectory
- final hidden base-vs-alt cosine and L2 ratio
- final hidden alt-vs-alt cosine
- aggregator alpha distribution
- residual/base norm ratio

Routing divergence alone is insufficient because different routes can still
produce nearly identical representations. If later-layer overlap rapidly returns
to near 1.0, final hidden cosine stays near 1.0, and aggregator residuals stay
near zero, the core hypothesis is unsupported. This is an important falsifiable
condition, not a failure to hide.

Recent path-centric MoE work is useful here: expert selections across layers
should be treated as paths, not independent one-layer events. Our method is not
PathMoE because it does not constrain router parameters, but it should be
evaluated with path-like metrics. Exact top-k set match is a stricter proxy than
fractional overlap for whether two trajectories are actually following the same
expert path.

## 8. Aggregator As A Small Latent Judge

Token-only fusion is underpowered for complex tasks because the judge may need
global problem context. The implemented aggregator therefore adds a masked mean
base-route representation to each token query:

```text
q_t = Wq LN(h_base,t) + Wg mean_t LN(h_base,t)
```

This keeps the aggregator small while allowing the same token position to choose
different trajectory evidence depending on the whole prompt/problem.

The aggregator also includes a null candidate by default:

```text
score_null,t = q_t^T k_null / sqrt(d)
v_null = 0
```

This matters because not every token should be forced to read a noisy
trajectory. The null candidate gives the latent judge a token-level abstention
choice while preserving the base residual anchor. If `null_alpha_mean` becomes
large, the learned adapter is saying that the alternative paths are often not
useful; if `alt_alpha_mass_mean` rises only on difficult tokens/prompts, it
supports the intended conditional-trajectory interpretation.

The value stream is base-relative by default:

```text
v_p,t = Wv (LN(h_p,t) - LN(h_0,t))
```

This is stricter than reading `LN(h_p,t)` directly. If an alternative trajectory
is identical to the base, it provides no correction evidence. The older absolute
value stream remains available as an ablation via `aggregator_value_mode`.

## 9. Metric Validity: Padding And Path Prefixes

The method lives or dies by whether the seed creates useful latent path
divergence. Therefore the metrics must measure the same mathematical object as
the claim.

Refinement implemented:

- Routing divergence metrics are masked by `attention_mask`.
- Layerwise top-k overlap and exact-match remain logged.
- A cumulative expert-path prefix exact-match is also logged:

```text
M_l^p(t) = 1[ S_i^p(t) = S_i^0(t) for every MoE layer i <= l ]
```

where `S_i^p(t)` is the unordered top-k expert set for token `t`, trajectory
`p`, and layer `i`.

This matters because layerwise agreement can recover after an early split.
For the research claim, the object is not only "which experts were selected at
layer l" but "which expert path did the token follow through the network".

## 10. Identity Initialization Versus Immediate Seed Learning

`Wo=0` guarantees:

```text
h_fused = h_base
logits_fused = logits_base
```

at initialization. This is mathematically desirable because the adapter starts
as the frozen base model. The tradeoff is that the first task-loss gradient does
not reach q/k/v or the seed noise:

```text
dL/dcontext = Wo^T dL/dh_fused = 0
```

The first update opens `Wo`; subsequent updates give task signal to the
aggregator attention, first-router seed, and context seed gate. This is
acceptable for a base-preserving adapter, but it should be considered when
choosing short training schedules. The seed diversity and L2 terms can move the
seed direction at step zero, but task-aligned seed and context-gate learning
begin after the residual path is nonzero.

## 11. Context-Independence Risk In MoE Routing

Open MoE analyses report that routing can be less context-sensitive than one
would hope, with assignments often learned early and correlated with token
identity. More recent layer-wise routing locality work refines this into a
layer-dependent claim: input layers may be more token-identity dominated, while
middle layers can show stronger context dependence. If the target model behaves
this way, a first-router seed may produce alternative token routes without
guaranteeing task-level perspectives.

Current mitigation:

- The aggregator query includes global base-route context, so fusion can depend
  on the whole problem.
- The first-router seed direction is trajectory-specific, but its strength is
  context-gated with identity initialization.
- Final hidden divergence is logged directly; route diversity alone is not taken
  as proof of useful representation diversity.
- Top-k margins and exact-match rates expose whether the seed is actually
  changing routes or merely perturbing logits below the decision boundary.

Refinement implemented:

```text
epsilon_p(x) = s_p * m_p(x) * center(noise_p)
m_p(x) = 1 + delta tanh(Wc LN(mean_valid hidden_base))
```

This preserves trajectory-specific seed directions while allowing the strength
of each candidate perspective to vary by query.

## 12. Expert Specialization Is Not Automatically Reasoning Perspective

DeepSeekMoE and OLMoE support the premise that sparse experts can specialize.
However, specialization can mean domain, vocabulary, syntax, or frequency
specialization rather than a distinct reasoning view of the same query.

Therefore the paper should avoid this overclaim:

```text
different expert combinations = different perspectives
```

and instead use this operational claim:

```text
different expert paths are candidate latent views; they count as useful
perspectives only if they produce measurable final-hidden complementarity and
improve fused CE over both the base model and the seed-disabled adapter.
```

This is why the implementation logs hidden divergence, path divergence,
aggregator alpha entropy/max, base CE delta, and includes the
`--disable_seed_noise` negative control.

## 13. Training Objective: Do Not Confuse Diversity With Usefulness

The central training risk is that route diversity can become an attractive but
irrelevant proxy. Forcing expert usage or route divergence directly can create
interference gradients: the router may satisfy the auxiliary objective while
hurting task likelihood or expert specialization. Recent MoE balancing work is
especially cautionary here: micro-batch balancing can over-uniformize domain
tokens, and loss-free balancing was proposed precisely because large auxiliary
losses can impair the main objective.

Refinement implemented:

```text
route/path divergence -> metric, not default loss
seed vector diversity -> small regularizer only
task likelihood       -> fused CE against labels
base preservation     -> KL(base || fused)
hidden safety         -> residual/base L2 ratio
```

This means the method is trained to predict next tokens and remain close to the
frozen base distribution, while diversity is used to diagnose the mechanism.

There is one optional exception: `--trajectory_oracle_aux_weight` adds a smooth
multiple-choice CE over noisy trajectory logits:

```text
L_oracle = softmin_p NLL(trajectory_p, gold), p > 0
```

This term is mathematically different from a route-divergence loss. It does not
reward different paths for being different; it rewards alternative hidden
states only when at least one of them better explains the gold token under the
frozen final norm and LM head. It is off by default because it increases memory
and can reduce complementarity if weighted too strongly.

The training code also supports target-only labels for paired datasets. This is
not a data-loader detail; it changes the scientific object. On instruction data,
training on prompt tokens would mostly optimize reconstruction of the question,
whereas target-only CE tests whether the seeded trajectories improve the answer
distribution.

The final claim should therefore be phrased as:

```text
The adapter learns to aggregate same-space trajectory representations that
contain complementary next-token evidence, as measured by fused CE, trajectory
oracle CE, and controlled KL/residual drift.
```

not:

```text
The adapter learns diverse routes.
```

## 14. Aggregator Judge Failure Is Distinct From Seed Failure

Recent latent/model-level aggregation work is aligned with our direction but
also exposes a sharper failure mode. MoT aggregates hidden states from frozen
heterogeneous peers through learned interaction layers, and DLLG learns
token-level fusion weights for expert logits. In both cases, the aggregation
module is not assumed to be correct just because experts are diverse; the
selector/fuser must learn when each expert is useful.

For our method this means:

```text
low trajectory oracle CE proves useful alternatives exist
low fused CE proves the aggregator exploited them
alpha-oracle alignment explains the gap
```

The implemented diagnostic compares token-level aggregator alpha against
per-trajectory gold-token NLL:

```text
best_alt_t = argmin_p>0 NLL_p,t
alpha_on_best_alt = alpha_t,best_alt_t
alt_oracle_regret = sum_p norm(alpha_p,t) NLL_p,t - min_p NLL_p,t
```

With the null candidate enabled, the base trajectory acts as the oracle target
for abstention:

```text
z_null,t = -NLL_base,t / tau
```

This gives a four-way interpretation:

- Route and hidden divergence are low: first-router seed is not controlling
  useful paths.
- Oracle CE is not below base CE: paths differ, but not in a predictive way.
- Oracle CE is below base CE but alpha ignores best alternatives: aggregator
  judge failure.
- Alpha follows oracle alternatives but fused CE does not improve: value stream,
  output projection, residual scale, or hidden-manifold constraints are the
  likely bottleneck.

The optional `--aggregator_oracle_align_weight` directly trains alpha toward a
soft oracle distribution over noisy trajectories and null:

```text
q*_t = stopgrad softmax([-NLL_1,t/tau ... -NLL_P,t/tau -NLL_base,t/tau])
L_align = CE(q*_t, alpha_t)
```

This is a useful stabilization ablation because `Wo=0` delays CE gradients to
the aggregator attention. It should not replace fused CE as the primary result;
it is a label-aware teacher for the latent judge.

Because `Wo=0` makes the initial fused logits exactly equal to the base logits,
the first optimizer step can put most task gradient on the aggregator output
projection while router-noise and q/k/v gradients remain small or zero. This is
acceptable only as a short identity-preserving warm start. Training logs should
therefore track group gradient norms:

```text
grad.aggregator.l2_norm
grad.router_noise.l2_norm
```

If router-noise gradients stay zero after the output projection has moved, the
method is not actually learning seed-conditioned trajectories; it is only
training a final residual adapter.

## 15. Fusion Conversion Is A Separate Scientific Object

The strongest version of the method is not:

```text
different routes exist
```

and not even:

```text
one noisy trajectory would have predicted the gold token better
```

The reported claim requires a conversion step:

```text
alternative advantage -> selective alpha mass -> residual value update
-> lower fused gold-token NLL
```

The code now logs this chain directly:

```text
alt_advantage_t = NLL_base,t - min_p>0 NLL_p,t
fusion_gain_t = NLL_base,t - NLL_fused,t
fusion_regret_t = NLL_fused,t - min_p>=0 NLL_p,t
```

This matters because recent confidence and ensemble work separates confidence
estimation from aggregation. LENS uses internal representations to learn
ensemble confidence; token-level ensembling work argues that ensembling should
be applied selectively; structural-confidence work treats hidden-state
trajectory stability as a correctness signal. In our setting, these imply that
trajectory diversity is only useful when the model can identify the tokens
where alternative hidden states contain better evidence and then move the
frozen LM-head distribution in that direction.

The key diagnostic split is:

- `alt_advantage_mean <= 0`: the seed did not create useful predictive
  alternatives.
- `alt_advantage_mean > 0` but alpha does not track advantage: latent judge
  failure.
- alpha tracks advantage but `fusion_improvement_mean <= 0`: residual value
  conversion failure.
- fused NLL improves only with large KL or residual norm: hidden-space rewrite,
  not a controlled latent ensemble.

This also makes selective abstention part of the method rather than a fallback.
When the base trajectory is already best, high null/base mass is correct. When
noisy alternatives have positive advantage, high alternative mass should
correlate with that advantage and with fused NLL improvement.

## 15.5 Heterogeneous-Ensemble Framing And Its Refinements

The honest framing of this method is not multi-agent debate: there is no
inter-trajectory communication round, so nothing is being "argued". The
correct claim is an amortized heterogeneous ensemble:

```text
One frozen MoE recovers part of the effect of querying several different
models by seeding several expert paths in a single batch-expanded forward
pass and fusing their final representations in latent space.
```

Compared to running N independent models or N token-space debate rounds, the
cost is O(N) latent passes with shared weights, and the fusion is a learned
pooling over same-space vectors rather than text-level voting. Four
refinements follow directly from taking that framing seriously:

1. Orthogonal seed initialization (`seed_init_mode=orthogonal`, default).
   If the seeds are meant to imitate different models, the candidate views
   should start maximally separated. Orthogonality inside the zero-sum
   expert-logit subspace is an initialization-time inductive bias; unlike a
   diversity loss it cannot fight task likelihood during training, which keeps
   the section-13 rule intact.

2. Frozen-random-seed control (`--freeze_seed_noise`). Random perturbation
   ensembles are a strong classical baseline, so "trained seeds" must beat
   "random fixed seeds + trained aggregator", not only "no seeds"
   (`--disable_seed_noise`). The two controls separate whether the
   perturbation must exist from whether it must be learned. If frozen random
   seeds match trained seeds, the seed-learning story collapses to noise
   ensembling with a learned judge.

3. Comparative judge keys (`aggregator_relative_keys`). A real ensemble judge
   weighs candidates against each other. Centering keys across the candidate
   axis makes the alt scores sum to zero per token, so only between-candidate
   differences carry selection signal and a direction shared by all noisy
   trajectories cannot buy alpha mass. Exposed as an ablation because it
   reduces the key space by one direction per token.

4. Advantage-weighted preservation (`--kl_advantage_tau`). Uniform
   `KL(base||fused)` is self-inconsistent with selective fusion: it pushes the
   fused distribution back to the base exactly on the tokens where the
   alternatives demonstrably contain better gold-token evidence. Weighting the
   per-token KL by `sigmoid(-alt_advantage/tau)` preserves where the base wins
   and yields where the alternatives win. The weight is stop-gradient and
   label-aware, so it has the same training-only status as the oracle
   auxiliaries and must not be applied at evaluation.

The evaluation consequence of the framing is that the target data should be
multi-domain and difficulty-skewed: a heterogeneous-ensemble effect predicts
the largest fused gains, alternative alpha mass, and realized seed scales on
the domains/tokens where the frozen base model is weakest. See
`docs/EXPERIMENT_DESIGN.md`.

## 16. References

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer", 2017:
  https://arxiv.org/abs/1701.06538
- Fedus et al., "Switch Transformers", 2021/2022:
  https://arxiv.org/abs/2101.03961
- Hazimeh et al., "DSelect-k: Differentiable Selection in the Mixture of
  Experts", NeurIPS 2021:
  https://arxiv.org/abs/2106.03760
- "Dense Backpropagation Improves Training for Sparse Mixture-of-Experts",
  2025:
  https://arxiv.org/html/2504.12463
- Gu et al., "Path-Constrained Mixture-of-Experts", 2026:
  https://arxiv.org/abs/2603.18297
- "Continuous Rerouting for Better Online Adaptation in Mixture of Experts",
  2025:
  https://arxiv.org/html/2510.14853
- Xue et al., "OpenMoE: An Early Effort on Open Mixture-of-Experts Language
  Models", 2024:
  https://arxiv.org/abs/2402.01739
- Muennighoff et al., "OLMoE: Open Mixture-of-Experts Language Models", 2024:
  https://arxiv.org/abs/2409.02060
- Hayashi et al., "Layer-wise MoE Routing Locality under Shared-Prefix Code
  Generation: Token-Identity Decomposition and Compile-Equivalent Fork
  Redundancy", 2026:
  https://arxiv.org/abs/2604.17182
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
  Fast LLM Ensembling", ICLR 2026 poster:
  https://openreview.net/forum?id=kfPkF2ACDM
- Xu et al., "Structural Confidence: Understanding and Diagnosing Hallucinations
  in Large Language Models", 2026:
  https://arxiv.org/html/2602.00977v1
- Wang et al., "The Myth of Expert Specialization in MoEs: Why Routing Reflects
  Geometry, Not Necessarily Domain Expertise", 2026:
  https://arxiv.org/html/2604.09780v1
- Guo et al., "Advancing Expert Specialization for Better MoE", NeurIPS 2025
  oral:
  https://openreview.net/forum?id=iydmH9boLb
- Li et al., "Expert Divergence Learning for MoE-based Language Models", 2026:
  https://arxiv.org/html/2603.00054v1
- Qiu et al., "Demons in the Detail: On Implementing Load Balancing Loss for
  Training Specialized Mixture-of-Expert Models", ACL 2025:
  https://aclanthology.org/2025.acl-long.249/
- "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", ICLR
  2025:
  https://openreview.net/forum?id=y1iU5czYpE
- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles", NeurIPS 2017:
  https://proceedings.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
- Wen et al., "BatchEnsemble", 2020:
  https://arxiv.org/abs/2002.06715
