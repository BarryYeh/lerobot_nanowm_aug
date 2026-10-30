#!/usr/bin/env bash
# early_anchors step 2: retrain the base policy with the OLD base's recipe (its train_config.json), from scratch,
# with denser checkpoints and W&B on, then evaluate every checkpoint with the shared eval protocol.
#   bash train_base.sh train         lerobot-train (GPU: $GPU, default 7)
#   bash train_base.sh eval          eval_success_steps.py on every checkpoint (skips finished ones);
#                                    EVAL_STEPS="010000 020000" (env) limits it to those checkpoints (one worker per GPU)
#   bash train_base.sh resume        continue an interrupted run from checkpoints/last (same W&B run)
#   bash train_base.sh tmux [resume] train (or resume) then eval, detached (tmux session early_anchors_base)
# Every setting comes from configs/early_anchors.yaml (base_train:, wandb:) and common.yaml (eval:).
# The pick of the checkpoint (success in base_train.target_success) is written to configs/base_policy_result.yaml.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false
RES=research/nanowm_aug
CFG=$RES/configs/early_anchors.yaml
y() { uv run python $RES/nwaug/attribution/config_io.py "$CFG" "$1"; }
GPU=${GPU:-7}
OUT=$(y base_train.output_dir); OUT=${OUT/#\~/$HOME}
LEROBOT_HOME=$(y lerobot_home)
EVAL_OUT=$(y out_dir)/base_eval

train() {
  [[ -e "$OUT" ]] && { echo "$OUT exists; delete it to retrain"; exit 1; }
  local entity=()
  case "$(y wandb.entity)" in ""|None|null) ;; *) entity=(--wandb.entity="$(y wandb.entity)") ;; esac
  CUDA_VISIBLE_DEVICES=$GPU uv run --extra training lerobot-train \
    --config_path="$(y base_train.recipe)" \
    --dataset.root="${LEROBOT_HOME/#\~/$HOME}/$(y datasets.real)" \
    --output_dir="$OUT" \
    --job_name=early_anchors_base_s"$(y base_train.seed)" \
    --steps="$(y base_train.steps)" \
    --save_freq="$(y base_train.save_freq)" \
    --seed="$(y base_train.seed)" \
    --policy.push_to_hub=false \
    --wandb.enable="$(y wandb.enable | tr 'TF' 'tf')" \
    --wandb.mode="$(y wandb.mode)" \
    --wandb.project="$(y wandb.project)" \
    "${entity[@]}"
}

resume() {
  # a checkpoint cut short by a failed save must be deleted first, so `last` points at a complete one
  CUDA_VISIBLE_DEVICES=$GPU uv run --extra training lerobot-train \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true --output_dir="$OUT"
}

eval_all() {
  mkdir -p "$EVAL_OUT"
  local cks=("$OUT"/checkpoints/[0-9]*)
  [[ -n ${EVAL_STEPS:-} ]] && { read -r -a cks <<< "$EVAL_STEPS"; cks=("${cks[@]/#/$OUT/checkpoints/}"); }
  for ck in "${cks[@]}"; do
    local step dest
    step=$(basename "$ck")
    dest=$EVAL_OUT/$step
    [[ -f $dest/success_steps.npz ]] && { echo "skip $step"; continue; }
    echo "eval $step -> $dest"
    CUDA_VISIBLE_DEVICES=$GPU uv run --extra training python $RES/nwaug/eval/eval_success_steps.py \
      --policy.path="$ck/pretrained_model" --policy.device=cuda --policy.use_amp=false \
      --policy.n_action_steps="$(y eval.n_action_steps)" --env.type=pusht \
      --eval.n_episodes="$(y eval.n_episodes)" --eval.batch_size="$(y eval.batch_size)" \
      --eval.use_async_envs=false --seed="$(y eval.seed)" --output_dir="$dest"
  done
}

case "${1:?usage: train_base.sh train|resume|eval|tmux [resume]}" in
  train) train ;;
  eval) eval_all ;;
  resume) resume ;;
  tmux) first=${2:-train}
        tmux new-session -d -s early_anchors_base "bash '$SELF' $first && bash '$SELF' eval; rc=\$?; echo; echo EXIT \$rc; sleep infinity"
        echo "tmux session early_anchors_base (attach: tmux a -t early_anchors_base)" ;;
  *) echo "unknown stage: $1"; exit 1 ;;
esac
