#!/usr/bin/env bash
# Task 2 -- image-only / n_obs=1 / fps 10 diffusion policy on the denoised 185-demo
# pusht_noise set (local/pushtnoise_clean185_imgonly, built by
# datagen/run_converter.py pushtnoise-denoise). The point: a policy whose behaviour stays in
# the pusht_noise regime so NanoWM can roll it out in the closed loop (GATE B).
# Hyperparams mirror proto_train_pusht.sh / lerobot/diffusion_pusht.
#
#   bash proto_train_pushtnoise.sh [SEED]
#   default: SEED=100000
set -euo pipefail

N_OBS=1
SEED="${1:-100000}"
HORIZON=16
N_ACTION=8
DROP=$(( HORIZON - N_ACTION - N_OBS + 1 ))   # = 8

REPO_ID=local/pushtnoise_clean185_imgonly
ROOT="${HF_LEROBOT_HOME:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/lerobot}/$REPO_ID"
OUT="outputs/train/pushtnoise_clean/obs${N_OBS}_s${SEED}"
LOGDIR="examples/nanowm_pusht_aug/logs"
LOG="$LOGDIR/train_pushtnoise_obs${N_OBS}_s${SEED}.log"
mkdir -p "$LOGDIR"

echo "n_obs_steps=$N_OBS seed=$SEED horizon=$HORIZON n_action_steps=$N_ACTION drop_n_last_frames=$DROP -> $OUT"

uv run --extra training lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.type=diffusion \
  --policy.n_obs_steps="$N_OBS" \
  --policy.horizon="$HORIZON" \
  --policy.n_action_steps="$N_ACTION" \
  --policy.drop_n_last_frames="$DROP" \
  --policy.crop_shape="[84, 84]" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --num_workers=8 \
  --steps=100000 \
  --save_freq=20000 \
  --env_eval_freq=0 \
  --seed="$SEED" \
  --output_dir="$OUT" \
  --job_name="proto_pushtnoise_clean_obs${N_OBS}_s${SEED}" 2>&1 | tee "$LOG"

echo "done -> $OUT/checkpoints/last/pretrained_model  (log: $LOG)"
