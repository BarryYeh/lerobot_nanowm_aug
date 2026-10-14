#!/usr/bin/env bash
# Base v2: diffusion policy on SD-VAE latents of the original PushT demos (local/pusht_latent).
#
# Same demos, fps 10, horizon / n_action_steps / drop / batch / steps / seed as the v1 RGB base
# (proto_train_pusht.sh -> ckpt.base); only the observation and its encoder differ:
#   v1  observation.image   (3,96,96)  RGB      -> resnet18 (ImageNet init), random crop 84/96
#   v2  observation.latent  (4,32,32)  SD-VAE   -> latent_conv (from scratch, GroupNorm), random crop 28/32
# The latent is declared VISUAL on the CLI (the dataset stores it as a float32 array, which lerobot
# would otherwise type as STATE); its stats.json entry is per VAE channel, (4,1,1).
# `--dataset.image_transforms.enable` is left off: it only touches camera (image/video) keys, of
# which this dataset has none.
#
#   bash train_latent_base.sh [N_OBS] [SEED]
#   defaults: N_OBS=1  SEED=100000
#   env: WANDB_ENABLE=true|false (default true; project nanowm_cl_aug, like finetune_cl.sh)
#
# The first run (obs1_s100000, 2026-10-07) predates the wandb flags; its 500 logged points (every
# 200 steps) were backfilled from the text log: wandb nanowm_cl_aug/runs/56nk6zbj (tag "backfill").
#
# Prereqs:
#   1. uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py pusht-latent --overwrite
#      (see datagen/convert_pusht_latent.py for the one-time zarr download)
#
# Eval (224px renders SD-VAE encoded in the loop; pairs with proto_eval_pusht.sh by episode order):
#   uv run --extra training python examples/nanowm_pusht_aug/eval/eval_latent_policy.py \
#     --policy.path="$OUT/checkpoints/last/pretrained_model" --policy.device=cuda --policy.use_amp=false \
#     --policy.n_action_steps=10 --env.type=pusht --eval.n_episodes=1000 --eval.batch_size=50 \
#     --eval.use_async_envs=false --seed=100000 --output_dir=<out>
set -euo pipefail

N_OBS="${1:-1}"
SEED="${2:-100000}"
HORIZON=16
N_ACTION=8
DROP=$(( HORIZON - N_ACTION - N_OBS + 1 ))   # LeRobot's drop_n_last_frames formula

REPO_ID=local/pusht_latent
ROOT="${HF_LEROBOT_HOME:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/lerobot}/$REPO_ID"
OUT="outputs/train/pusht_latent/obs${N_OBS}_s${SEED}"
LOGDIR="examples/nanowm_pusht_aug/logs"
LOG="$LOGDIR/train_latent_obs${N_OBS}_s${SEED}.log"
mkdir -p "$LOGDIR"   # NOT $OUT -- lerobot-train refuses to start if output_dir exists

echo "n_obs_steps=$N_OBS seed=$SEED horizon=$HORIZON n_action_steps=$N_ACTION drop_n_last_frames=$DROP -> $OUT"

uv run --extra training lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --policy.type=diffusion \
  --policy.input_features='{"observation.latent": {"type": "VISUAL", "shape": [4, 32, 32]}}' \
  --policy.vision_backbone=latent_conv \
  --policy.pretrained_backbone_weights=null \
  --policy.n_obs_steps="$N_OBS" \
  --policy.horizon="$HORIZON" \
  --policy.n_action_steps="$N_ACTION" \
  --policy.drop_n_last_frames="$DROP" \
  --policy.crop_shape="[28, 28]" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable="${WANDB_ENABLE:-true}" \
  --wandb.mode=online \
  --wandb.project=nanowm_cl_aug \
  --batch_size=64 \
  --num_workers=8 \
  --steps=100000 \
  --save_freq=20000 \
  --env_eval_freq=0 \
  --seed="$SEED" \
  --output_dir="$OUT" \
  --job_name="pusht_latent_obs${N_OBS}_s${SEED}" 2>&1 | tee "$LOG"

echo "done -> $OUT/checkpoints/last/pretrained_model  (log: $LOG)"
