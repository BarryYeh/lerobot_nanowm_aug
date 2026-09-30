#!/usr/bin/env bash
# Fine-tune one arm of the closed-loop augmentation comparison.
#   arm A (real-only) : bash finetune_cl.sh a
#   arm B (real+aug)  : bash finetune_cl.sh b
#   any dataset       : bash finetune_cl.sh repo=local/<repo_id>   (tag = repo id minus 'local/')
# SAVE_FREQ (env, default 10000): 20000 keeps only the final checkpoint (~2 GB instead of ~6 GB).
# BOTH arms are identical except for the dataset: same pretrained checkpoint, same seed,
# same LR, same steps, and the same drop_n_last_frames (4th arg, REQUIRED). drop MUST equal
# the packer's --ep-len minus 1 so each imagined episode yields exactly ONE sample: default
# ep-len 10 -> drop 9 (-0.9% real samples vs the baseline's 8). Pass 8 for ep-len-9 datasets.
# (The earlier 16-frame layout forced drop=15, which silently stripped the 1442 highest-coverage
# real observation frames -- mean coverage 0.608 vs 0.258 kept -- from BOTH arms.)
# The aggregate's normalization stats were pinned to the real-only set, so VISUAL MEAN_STD
# normalization is identical in both arms as well.
set -euo pipefail
ARM="${1:?usage: finetune_cl.sh <arm|repo=local/<id>> <seed> <steps> <drop>  (drop = pack --ep-len minus 1)}"
SEED="${2:-1000}"
STEPS="${3:-20000}"
DROP="${4:?usage: finetune_cl.sh <arm|repo=local/<id>> <seed> <steps> <drop>  (drop = pack --ep-len minus 1)}"
[[ "$DROP" =~ ^[0-9]+$ ]] || { echo "drop must be a non-negative integer"; exit 1; }
CKPT=outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model
case "$ARM" in
  a) REPO_ID=local/pusht_imgonly;         TAG=a_real_only ;;
  b) REPO_ID=local/pusht_imgonly_plus_cl9; TAG=b_real_plus_aug ;;
  c) REPO_ID=local/pusht_plus_oracle;      TAG=c_oracle ;;
  d) REPO_ID=local/pusht_plus_trunc95;     TAG=d_trunc95 ;;
  untrimmed) REPO_ID=local/pusht_plus_untrimmed464; TAG=untrimmed464 ;;
  trimmed)   REPO_ID=local/pusht_plus_trimmed464;   TAG=trimmed464 ;;
  verified)  REPO_ID=local/pusht_plus_verified464;  TAG=verified464 ;;
  verified_coh) REPO_ID=local/pusht_plus_verified464_coh; TAG=verified464_coh ;;
  trimmed_coh)  REPO_ID=local/pusht_plus_trimmed464_coh;  TAG=trimmed464_coh ;;
  lastpush)     REPO_ID=local/pusht_plus_lastpush464;     TAG=lastpush464 ;;
  allpre)       REPO_ID=local/pusht_plus_allpre464;       TAG=allpre464 ;;
  window)       REPO_ID=local/pusht_plus_window464;       TAG=window464 ;;
  untrimmed_h)  REPO_ID=local/pusht_plus_untrimmed_h10;   TAG=untrimmed_h10 ;;
  allpre_h)     REPO_ID=local/pusht_plus_allpre_h10;      TAG=allpre_h10 ;;
  windowv)      REPO_ID=local/pusht_plus_windowv464;      TAG=windowv464 ;;
  untrimmed500) REPO_ID=local/pusht_plus_untrimmed500; TAG=untrimmed500 ;;
  trimmed500)   REPO_ID=local/pusht_plus_trimmed500;   TAG=trimmed500 ;;
  verified500)  REPO_ID=local/pusht_plus_verified500;  TAG=verified500 ;;
  repo=*)       REPO_ID="${ARM#repo=}";                  TAG="${REPO_ID#local/}" ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac
ROOT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$REPO_ID"
OUT="outputs/train/cl_aug/${TAG}_d${DROP}_s${SEED}"
mkdir -p examples/nanowm_pusht_aug/logs
echo "arm=$ARM dataset=$REPO_ID seed=$SEED steps=$STEPS -> $OUT"
uv run --extra training lerobot-train \
  --policy.path="$CKPT" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.drop_n_last_frames="$DROP" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-5 \
  --wandb.enable=true \
  --wandb.mode=online \
  --wandb.project=nanowm_cl_aug \
  --batch_size=64 \
  --num_workers=4 \
  --steps="$STEPS" \
  --save_freq="${SAVE_FREQ:-10000}" \
  --env_eval_freq=0 \
  --seed="$SEED" \
  --output_dir="$OUT"
