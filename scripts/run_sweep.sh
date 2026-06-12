#!/usr/bin/env bash
# Hyperparameter sweep on the FIXED main_v1 data mix (no data changes).
# Diagnosis-driven queue: judge strength first (alpha_on_best ~0.33 = near
# uniform), then seed strength, KL freedom, longer training, then N=5.
# Each config: train -> fast eval (gsm8k+svamp @100, arc_challenge+openbookqa
# @300) against the existing main_v1_base references.
# Run INSIDE the container:
#   docker compose ... run --rm moe-aggregate bash scripts/run_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MIX="data/train/main_v2_mix.jsonl"
FAST_MATH="gsm8k,svamp"
FAST_CHOICE="arc_challenge,openbookqa"

# name|num_traj|extra train args
CONFIGS=(
  "v2_judge|3|--aggregator_oracle_align_weight 0.1 --aggregator_relative_keys --steps 2500"
  "v3_judge_strong|3|--aggregator_oracle_align_weight 0.3 --aggregator_relative_keys --trajectory_oracle_aux_weight 0.05 --steps 2500"
  "v4_seed|3|--aggregator_oracle_align_weight 0.1 --aggregator_relative_keys --noise_scale 0.45 --steps 2500"
  "v5_free|3|--aggregator_oracle_align_weight 0.1 --aggregator_relative_keys --kl_beta_end 0.02 --steps 4000"
  "v6_n5|5|--aggregator_oracle_align_weight 0.1 --aggregator_relative_keys --steps 2500"
  "v7_alllayer|3|--seed_inject_mode all --aggregator_oracle_align_weight 0.1 --aggregator_relative_keys --steps 2500"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  rest="${entry#*|}"
  traj="${rest%%|*}"
  extra="${rest#*|}"
  echo "==== [sweep:${name}] train (N=${traj}) ===="
  if [ ! -f "outputs/${name}/trajectory_adapter.pt" ]; then
    # shellcheck disable=SC2086
    python scripts/train_adapter.py \
      --train_file "${MIX}" \
      --num_trajectories "${traj}" \
      --batch_size 1 \
      --max_length 1024 \
      --noise_init_std 0.5 \
      --noise_scale 0.3 \
      --noise_l2_weight 1e-5 \
      --kl_beta_start 1.0 \
      --kl_beta_end 0.05 \
      --kl_advantage_tau 0.5 \
      --trajectory_oracle_aux_weight 0.02 \
      --log_every 10 \
      --out_dir "outputs/${name}" \
      ${extra} || { echo "==== [sweep:${name}] TRAIN FAILED ===="; continue; }
  fi
  echo "==== [sweep:${name}] fast eval ===="
  if [ ! -f "results/${name}_math/metrics.json" ]; then
    python scripts/evaluate.py --mode fused --adapter "outputs/${name}/trajectory_adapter.pt" \
      --num_trajectories "${traj}" --tasks "${FAST_MATH}" \
      --limit 100 --batch_size 8 --out_dir "results/${name}_math" \
      || echo "==== [sweep:${name}] MATH EVAL FAILED ===="
  fi
  if [ ! -f "results/${name}_choice/metrics.json" ]; then
    python scripts/evaluate.py --mode fused --adapter "outputs/${name}/trajectory_adapter.pt" \
      --num_trajectories "${traj}" --tasks "${FAST_CHOICE}" \
      --limit 300 --score_batch 16 --out_dir "results/${name}_choice" \
      || echo "==== [sweep:${name}] CHOICE EVAL FAILED ===="
  fi
  echo "==== [sweep:${name}] done ===="
done

echo "==== sweep complete ===="
for name in v2_judge v3_judge_strong v4_seed v5_free v6_n5 v7_alllayer; do
  for kind in math choice; do
    f="results/${name}_${kind}/metrics.json"
    [ -f "$f" ] && python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for m in d["metrics"]:
    extra = f" norm={m['accuracy_norm']:.3f}" if "accuracy_norm" in m else ""
    print(f"{sys.argv[1]}: {m['task']} acc={m.get('accuracy', 0):.3f}{extra}")
PY
  done
done
