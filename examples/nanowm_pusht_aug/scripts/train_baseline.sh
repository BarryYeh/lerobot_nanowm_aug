#!/usr/bin/env bash
# Task #1 -- image-only diffusion policy baseline ("a") on the fps-2 pusht_noise data.
#
#   bash train_baseline.sh [SEED] [N_OBS_STEPS]
#   defaults: SEED=100000  N_OBS_STEPS=2
#
# n_obs_steps=1 was an over-correction (see README): once the real data is also
# fps 2, n_obs_steps=2 is fully consistent with the fps-2 imagined episodes and
# gives the policy the velocity/history it needs for closed-loop control.
# The fps-2 (frame_interval=5) sub-sampling IS the NanoWM alignment; n_obs_steps is free.
#
# Prereqs:
#   1. Patched modeling_diffusion.py (observation.state optional).
#   2. uv run python examples/nanowm_pusht_aug/datagen/run_converter.py pusht-noise \
#          --splits train val --frame-interval 5 --drop-state --overwrite
#      -> local/pusht_noise_224_fi5_imgonly
#
# Same hyperparams must be reused for "a+aug".
set -euo pipefail

SEED="${1:-100000}"
N_OBS="${2:-2}"
HORIZON=16
N_ACTION=8
DROP=$(( HORIZON - N_ACTION - N_OBS + 1 ))   # LeRobot's drop_n_last_frames formula

REPO_ID=local/pusht_noise_224_fi5_imgonly
ROOT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$REPO_ID"
OUT="outputs/train/nanowm_aug/baseline_a_obs${N_OBS}_s${SEED}"

echo "seed=$SEED n_obs_steps=$N_OBS horizon=$HORIZON n_action_steps=$N_ACTION drop_n_last_frames=$DROP -> $OUT"

uv run --extra training lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.type=diffusion \
  --policy.n_obs_steps="$N_OBS" \
  --policy.horizon="$HORIZON" \
  --policy.n_action_steps="$N_ACTION" \
  --policy.drop_n_last_frames="$DROP" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --num_workers=8 \
  --steps=100000 \
  --save_freq=20000 \
  --env_eval_freq=0 \
  --seed="$SEED" \
  --output_dir="$OUT" \
  --job_name="nanowm_aug_baseline_a_obs${N_OBS}_s${SEED}"

# Eval (2 Hz control loop, native 224 render):
#   uv run --extra training python examples/nanowm_pusht_aug/eval/eval_baseline.py \
#     "$OUT/checkpoints/last/pretrained_model" --n-episodes 100 --seed "$SEED"
