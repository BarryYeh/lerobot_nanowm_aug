#!/usr/bin/env bash
# early_anchors step 7: fine-tune the Img and Real arms from the base, evaluate them and the base, report.
# Criterion: <out_dir>/criteria.md (fixed 2026-10-21 before any fine-tune).
#   bash run_finetune.sh finetune   2 arms x finetune_seeds, scripts/finetune_cl.sh (20k steps, drop 8), all at once on $GPU
#   bash run_finetune.sh eval       the 4 fine-tunes + the base, 2000 paired episodes (seed 100000, na 10), all at once
#   bash run_finetune.sh report     report.py -> <out_dir>/finetune/eval/report.{json,md}
#   bash run_finetune.sh tmux       finetune, eval, report, detached (tmux session early_anchors_ft)
# W&B project / mode come from configs/early_anchors.yaml (wandb:). Stages skip finished runs.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false
RES=research/nanowm_aug
CFG=$RES/configs/early_anchors.yaml
y() { uv run python $RES/nwaug/attribution/config_io.py "$CFG" "$1"; }
GPU=${GPU:-0}  # this node's shell only sees GPU 0
OUT=$(y out_dir); OUT=${OUT/#\~/$HOME}
FT=$OUT/finetune
BASE_CKPT=$(uv run python -c "import yaml; print(yaml.safe_load(open('$RES/configs/base_policy_result.yaml'))['early_anchors_base']['ckpt'])")
TAG=ea_f2n10
ARMS=(img real)
SEEDS=(1000 1001)
STEPS=20000
DROP=8
N_EVAL=2000
NA_EVAL=10  # rollout f=2 == na 10; early_anchors.yaml's eval.n_action_steps (5) is NOT used here

run_dir() { echo "$FT/pusht_${TAG}_$1_balanced_d${DROP}_s$2"; }

finetune() {
  mkdir -p "$FT/logs"
  local pids=()
  for arm in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do
    [[ -d $(run_dir "$arm" "$s")/checkpoints/last ]] && { echo "skip $arm s$s"; continue; }
    rm -rf "$(run_dir "$arm" "$s")"
    echo "finetune $arm s$s -> $(run_dir "$arm" "$s")"
    CUDA_VISIBLE_DEVICES=$GPU BASE_CKPT=$BASE_CKPT OUT_ROOT=$FT SAVE_FREQ=$STEPS \
      WANDB_PROJECT=$(y wandb.project) WANDB_MODE=$(y wandb.mode) WANDB_ENABLE=$(y wandb.enable | tr 'TF' 'tf') \
      bash $RES/proj/pusht/scripts/finetune_cl.sh "repo=local/pusht_${TAG}_${arm}_balanced" "$s" $STEPS $DROP \
      > "$FT/logs/ft_${arm}_s$s.log" 2>&1 &
    pids+=($!)
  done; done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

eval_one() {  # <ckpt> <dest>
  [[ -f $2/success_steps.npz ]] && { echo "skip eval $(basename "$2")"; return 0; }
  mkdir -p "$2"
  CUDA_VISIBLE_DEVICES=$GPU uv run --extra training python $RES/nwaug/eval/eval_success_steps.py \
    --policy.path="$1" --policy.device=cuda --policy.use_amp=false --policy.n_action_steps=$NA_EVAL \
    --env.type=pusht --eval.n_episodes=$N_EVAL --eval.batch_size="$(y eval.batch_size)" \
    --eval.use_async_envs=false --seed="$(y eval.seed)" --output_dir="$2" > "$2.log" 2>&1
}

eval_all() {
  mkdir -p "$FT/eval"
  local pids=()
  eval_one "$BASE_CKPT" "$FT/eval/base" & pids+=($!)
  for arm in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do
    eval_one "$(run_dir "$arm" "$s")/checkpoints/last/pretrained_model" "$FT/eval/${arm}_s$s" & pids+=($!)
  done; done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

report() { uv run python $RES/proj/pusht/early_anchors/report.py "$FT/eval" --seeds "${SEEDS[@]}"; }

case "${1:?usage: run_finetune.sh finetune|eval|report|tmux}" in
  finetune) finetune ;;
  eval) eval_all ;;
  report) report ;;
  tmux) tmux new-session -d -s early_anchors_ft \
          "bash '$SELF' finetune && bash '$SELF' eval && bash '$SELF' report; rc=\$?; echo; echo EXIT \$rc; sleep infinity"
        echo "tmux session early_anchors_ft (attach: tmux a -t early_anchors_ft)" ;;
  *) echo "unknown stage: $1"; exit 1 ;;
esac
