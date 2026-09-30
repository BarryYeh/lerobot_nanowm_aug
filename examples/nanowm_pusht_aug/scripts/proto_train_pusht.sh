#!/usr/bin/env bash
# Proto: image-only diffusion policy on lerobot/pusht (native fps 10), varying n_obs_steps.
#
# Purpose: check whether the earlier n_obs_steps=1 failure was a dataset problem
# (pusht_noise has no goal-directed trajectories) rather than n_obs_steps itself.
# Hyperparams mirror lerobot/diffusion_pusht, except this is image-only (no
# observation.state -- the dataset local/pusht_imgonly carries none, so the policy
# factory derives an image-only policy on its own).
#
#   bash proto_train_pusht.sh [N_OBS] [SEED]
#   defaults: N_OBS=1  SEED=100000
#
# Prereqs:
#   1. Patched modeling_diffusion.py / configuration_diffusion.py (observation.state optional).
#   2. uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py pusht-imgonly --overwrite
set -euo pipefail

N_OBS="${1:-1}"
SEED="${2:-100000}"
HORIZON=16
N_ACTION=8
DROP=$(( HORIZON - N_ACTION - N_OBS + 1 ))   # LeRobot's drop_n_last_frames formula

REPO_ID=local/pusht_imgonly
ROOT="${HF_LEROBOT_HOME:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/lerobot}/$REPO_ID"
OUT="outputs/train/pusht_imgonly/obs${N_OBS}_s${SEED}"
LOGDIR="examples/nanowm_pusht_aug/logs"
LOG="$LOGDIR/train_obs${N_OBS}_s${SEED}.log"
mkdir -p "$LOGDIR"   # NOT $OUT -- lerobot-train refuses to start if output_dir exists

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
  --job_name="proto_pusht_imgonly_obs${N_OBS}_s${SEED}" 2>&1 | tee "$LOG"

echo "done -> $OUT/checkpoints/last/pretrained_model  (log: $LOG)"
