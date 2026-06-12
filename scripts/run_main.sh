#!/usr/bin/env bash
# Main experiment: POC diagnostics showed two cold-start failures, fixed here.
#   1) seed perturbation (scale*init_std ~ 0.002) was ~50x below router logit
#      margins (~0.1-0.15) -> no route flips -> noise_init_std 0.5, scale 0.3
#   2) Wo=0 blocked task credit to judge/seeds -> enable the documented
#      warm-start auxiliaries (oracle align + trajectory oracle aux)
# Run INSIDE the container:
#   docker compose -f docker/docker-compose.yml run --rm moe-aggregate bash scripts/run_main.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-2000}"
MATH_LIMIT="${MATH_LIMIT:-100}"
CHOICE_LIMIT="${CHOICE_LIMIT:-300}"
TRAJ="${TRAJ:-3}"
RUN="${RUN:-main_v1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-38000}"
EVAL_TASKS="gsm8k,svamp,multiarith,addsub,singleeq"
CHOICE_TASKS="arc_easy,arc_challenge,openbookqa,siqa,hellaswag"

echo "== [0/5] GPU check =="
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
echo "total free GPU memory: ${free_mb} MiB (need >= ${GPU_MIN_FREE_MB})"
if [ "${free_mb}" -lt "${GPU_MIN_FREE_MB}" ]; then
  echo "GPUs are busy; rerun when total free memory >= ${GPU_MIN_FREE_MB} MiB." >&2
  exit 1
fi

echo "== [1/5] training mix (hard prefixes oversampled 5x) =="
if [ ! -f "data/train/${RUN}_mix.jsonl" ]; then
  for _ in 1 2 3 4 5; do cat data/train/stage2_hard_prefix.jsonl; done > "data/train/${RUN}_mix.jsonl"
  cat data/train/stage3_sft.jsonl >> "data/train/${RUN}_mix.jsonl"
fi
wc -l "data/train/${RUN}_mix.jsonl"

echo "== [2/5] adapter training (${STEPS} steps) =="
if [ ! -f "outputs/${RUN}/trajectory_adapter.pt" ]; then
  python scripts/train_adapter.py \
    --train_file "data/train/${RUN}_mix.jsonl" \
    --num_trajectories "${TRAJ}" \
    --steps "${STEPS}" \
    --batch_size 1 \
    --max_length 512 \
    --noise_init_std 0.5 \
    --noise_scale 0.3 \
    --noise_l2_weight 1e-5 \
    --kl_beta_start 1.0 \
    --kl_beta_end 0.05 \
    --kl_advantage_tau 0.5 \
    --aggregator_oracle_align_weight 0.02 \
    --trajectory_oracle_aux_weight 0.02 \
    --log_every 10 \
    --out_dir "outputs/${RUN}"
fi

echo "== [3/5] eval: base reference =="
if [ ! -f "results/${RUN}_base/metrics.json" ]; then
  python scripts/evaluate.py --mode base --tasks "${EVAL_TASKS}" \
    --limit "${MATH_LIMIT}" --out_dir "results/${RUN}_base"
  python scripts/evaluate.py --mode base --tasks "${CHOICE_TASKS}" \
    --limit "${CHOICE_LIMIT}" --out_dir "results/${RUN}_base_choice"
fi

echo "== [4/5] eval: fused adapter + no-seed control =="
if [ ! -f "results/${RUN}_fused/metrics.json" ]; then
  python scripts/evaluate.py --mode fused --adapter "outputs/${RUN}/trajectory_adapter.pt" \
    --num_trajectories "${TRAJ}" --tasks "${EVAL_TASKS}" \
    --limit "${MATH_LIMIT}" --out_dir "results/${RUN}_fused"
  python scripts/evaluate.py --mode fused --adapter "outputs/${RUN}/trajectory_adapter.pt" \
    --num_trajectories "${TRAJ}" --tasks "${CHOICE_TASKS}" \
    --limit "${CHOICE_LIMIT}" --out_dir "results/${RUN}_fused_choice"
fi

echo "== [5/5] trace for the viewer =="
python scripts/generate_traced.py \
  --adapter "outputs/${RUN}/trajectory_adapter.pt" \
  --num_trajectories "${TRAJ}" \
  --prompt "A store had 120 apples. They sold 45 in the morning and got a delivery of 30 more in the afternoon. How many apples does the store have now?" \
  --max_new_tokens 128 \
  --out "viz/traces/${RUN}.json"

echo "done ${RUN}. compare results/${RUN}_base*/metrics.json vs results/${RUN}_fused*/metrics.json"
