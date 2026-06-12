# MoE Aggregate

Seeded Latent Trajectory Aggregation for `deepseek-ai/deepseek-moe-16b-chat`.

The implementation keeps the DeepSeek-MoE backbone frozen, injects trainable
trajectory seed noise only into the first MoE router, then fuses the final
pre-norm hidden states with a small base-anchored residual aggregator.
The seed direction is trajectory-specific; by default an identity-initialized
bounded context gate lets the prompt adjust only the first-router seed strength.
Seed rows start pairwise-orthogonal in the zero-sum expert-logit subspace
(`--seed_init_mode orthogonal`), so candidate latent views begin maximally
separated without any diversity loss. The aggregator also has a null candidate,
letting it abstain from noisy trajectories token-by-token, and reads
base-relative trajectory deltas by default; `--aggregator_relative_keys`
switches the judge to comparative (candidate-centered) key scoring.

## Docker

This project reuses the existing `moe-p:latest` image and the existing
`docker_hf-cache` volume.

```bash
docker compose -f docker/docker-compose.yml run --rm moe-aggregate
```

Inside the container:

```bash
python scripts/check_env.py
python tests/test_math_components.py
python scripts/smoke_forward.py --num_trajectories 3
python scripts/smoke_forward.py --num_trajectories 5
python scripts/smoke_generate.py --num_trajectories 3 --max_new_tokens 4
```

The smoke scripts load the cached DeepSeek-MoE checkpoint with
`local_files_only=True` by default.

Data, evaluation, and the end-to-end POC:

```bash
python scripts/prepare_eval_data.py --out_dir data/eval        # RoE-matched 12 tasks
python scripts/prepare_train_data.py --out_dir data/train --eval_dir data/eval
python scripts/mine_hard_prefixes.py --pool data/train/stage1_continuation.jsonl
python scripts/evaluate.py --mode base  --tasks math --limit 100
python scripts/evaluate.py --mode fused --adapter outputs/adapter/trajectory_adapter.pt
bash scripts/run_poc.sh   # data -> mine -> train -> eval -> trace, with GPU check
```

`scripts/decontamination.py` builds n-gram filters from `data/eval/*.jsonl` so
adapter training data can be checked against the evaluation suite; see
[data/README.md](data/README.md) for the mixture design and
[results comparison] in `results/*/metrics.json` after evaluation runs.

Per-token trajectory visualization:

```bash
python scripts/generate_traced.py --prompt "..." --out viz/traces/trace.json
python -m http.server -d viz 8000   # open http://localhost:8000/?trace=traces/trace.json
```

The viewer colors generated tokens by noisy-trajectory usage; clicking a token
opens the forward position that produced it: aggregator alpha (including null
abstention), per-trajectory top-1 predictions, final-hidden cosine to base, and
a layer-by-expert grid of every trajectory's top-k routing path with
base-divergent layers and experts highlighted.

Adapter training:

```bash
python scripts/train_adapter.py \
  --train_file data/train.jsonl \
  --num_trajectories 3 \
  --steps 100
```

`train_adapter.py` trains only the first-router seed noise and the final
aggregator. `--kl_advantage_tau` optionally weights the per-token base KL by
trajectory advantage, preserving the base distribution where it wins and
relaxing it where alternatives carry better gold-token evidence.
`--freeze_seed_noise` keeps random fixed seeds active but trains only the
aggregator (the frozen-random-seed negative control), while
`--disable_seed_noise` removes the perturbation entirely. JSONL supports `text`, `prompt+completion`, `question+answer`,
`instruction+output`, and chat `messages`. Paired examples use target-only
labels by default: prompt tokens are masked with `-100`, and only answer tokens
contribute next-token CE. The loss is fused CE plus scheduled base KL, a small
hidden residual penalty, and optional noisy-trajectory soft-oracle CE via
`--trajectory_oracle_aux_weight`. The logger also compares aggregator alpha to
trajectory gold-token NLLs, so runs can distinguish bad trajectory seeds from a
latent judge that ignores useful alternatives. `--aggregator_oracle_align_weight`
can optionally train that alpha judge toward a soft oracle over noisy
trajectories plus the null/base abstention candidate. Additional diagnostics
measure whether alternative trajectory advantage is actually converted into
lower fused gold-token NLL, separating seed failure, judge failure, and
residual-value conversion failure. Training logs also include router-noise and
aggregator gradient norms, which is useful because the identity-initialized
output projection can intentionally delay seed/router gradients during the
first optimizer steps.

`tests/test_math_components.py` does not load DeepSeek weights. It checks the
core math on tiny toy modules: identity-initialized aggregation, centered seed
noise, context-gated seed strength, first-router-only injection, route/path
metrics, prompt/target label masking, trajectory prediction diagnostics, and
gradient flow into the seed noise.

See [docs/METHOD.md](docs/METHOD.md) for the method invariants and
[docs/METHOD_AUDIT.md](docs/METHOD_AUDIT.md) for the mathematical failure
modes and refinements. [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md)
defines the falsifiable evidence chain for the research claim.
