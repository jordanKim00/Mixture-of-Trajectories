#!/usr/bin/env bash
# End-to-end POC: data -> mining -> training -> RoE-matched eval -> trace.
# Run INSIDE the container:
#   docker compose -f docker/docker-compose.yml run --rm moe-aggregate bash scripts/run_poc.sh
# Tune with env vars: STEPS, LIMIT, TRAJ, GPU_MIN_FREE_MB.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-300}"
LIMIT="${LIMIT:-100}"
TRAJ="${TRAJ:-3}"
# deepseek-moe-16b in bf16 is ~33GB of weights sharded by device_map=auto,
# so gate on TOTAL free memory across cards, not a single card.
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-38000}"

echo "== [0/7] GPU check =="
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
echo "total free GPU memory: ${free_mb} MiB (need >= ${GPU_MIN_FREE_MB})"
if [ "${free_mb}" -lt "${GPU_MIN_FREE_MB}" ]; then
  echo "GPUs are busy; rerun when total free memory >= ${GPU_MIN_FREE_MB} MiB." >&2
  exit 1
fi

echo "== [1/7] sanity tests =="
python tests/test_math_components.py

echo "== [2/7] eval data =="
[ -f data/eval/gsm8k.jsonl ] || python scripts/prepare_eval_data.py

echo "== [3/7] train data =="
[ -f data/train/stage3_sft.jsonl ] || python scripts/prepare_train_data.py

echo "== [4/7] stage-2 hard-prefix mining =="
if [ ! -f data/train/stage2_hard_prefix.jsonl ]; then
  python scripts/mine_hard_prefixes.py \
    --pool data/train/stage1_continuation.jsonl \
    --limit 2000 --keep_fraction 0.25
fi

echo "== [5/7] adapter training (stage3 SFT on hard-mixed data) =="
if [ ! -f outputs/poc_adapter/trajectory_adapter.pt ]; then
  cat data/train/stage2_hard_prefix.jsonl data/train/stage3_sft.jsonl > data/train/poc_mix.jsonl
  python scripts/train_adapter.py \
    --train_file data/train/poc_mix.jsonl \
    --num_trajectories "${TRAJ}" \
    --steps "${STEPS}" \
    --batch_size 1 \
    --max_length 512 \
    --out_dir outputs/poc_adapter
fi

echo "== [6/7] RoE-matched eval (limit ${LIMIT}/task) =="
python scripts/evaluate.py --mode base  --tasks gsm8k,svamp,arc_easy,arc_challenge,openbookqa \
  --limit "${LIMIT}" --out_dir results/poc_base
python scripts/evaluate.py --mode fused --adapter outputs/poc_adapter/trajectory_adapter.pt \
  --num_trajectories "${TRAJ}" --tasks gsm8k,svamp,arc_easy,arc_challenge,openbookqa \
  --limit "${LIMIT}" --out_dir results/poc_fused
python scripts/evaluate.py --mode fused --disable_seed_noise \
  --num_trajectories "${TRAJ}" --tasks gsm8k,svamp,arc_easy,arc_challenge,openbookqa \
  --limit "${LIMIT}" --out_dir results/poc_fused_noseed

echo "== [7/7] trajectory trace for the viewer =="
python scripts/generate_traced.py \
  --adapter outputs/poc_adapter/trajectory_adapter.pt \
  --num_trajectories "${TRAJ}" \
  --prompt "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at \$2 per egg. How much does she make daily?" \
  --max_new_tokens 128 \
  --out viz/traces/poc.json

echo "done. compare results/poc_*/metrics.json ; view trace:"
echo "  python -m http.server -d viz 8000  ->  http://localhost:8000/?trace=traces/poc.json"
