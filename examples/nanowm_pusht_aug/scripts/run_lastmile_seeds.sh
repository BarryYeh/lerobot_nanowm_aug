#!/usr/bin/env bash
# Last-mile seed replication (criteria.md addenda 1-2): fine-tune seed 1001 for W, R and ctrl on the SAME
# datasets, evaluate each on the same 2000 paired episodes, pool with seed 1000 (the existing run).
#   bash run_lastmile_seeds.sh tmux    detached; GPUS="6 7" by default
# Resumable: finished stages leave <out>/.done_seeds/<stage>; progress in <out>/status_seeds.txt.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
P=examples/nanowm_pusht_aug
PY="uv run --extra training python"
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
O=$HOME/datastorage/nanowm_pusht_aug/lastmile

if [ "${1:-}" = "tmux" ]; then
  tmux has-session -t lastmile_seeds 2>/dev/null && { echo "tmux session lastmile_seeds already running"; exit 1; }
  tmux new-session -d -s lastmile_seeds "GPUS='${GPU_LIST[*]}' WANDB_ENABLE='${WANDB_ENABLE:-true}' bash '$SELF' 2>&1 | tee -a '$O/run_seeds.log'"
  echo "started tmux session lastmile_seeds; log $O/run_seeds.log, status $O/status_seeds.txt"
  exit 0
fi

mkdir -p "$O/.done_seeds" "$O/logs"
status() { local l; l="$(date '+%F %T') $*"; echo "$l" >> "$O/status_seeds.txt"; echo "$l" 2>/dev/null || true; }
FINISHED=0
trap '[ "$FINISHED" = 1 ] || status "FAIL driver exited unexpectedly (code $?); re-run to resume"' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM

UNITS=("W 1001" "R 1001" "ctrl 1001")   # criteria.md addendum 2: 2 seeds (1000 existing + 1001)
repo_of() { case $1 in ctrl) echo local/pusht_imgonly ;; *) echo "local/pusht_lastmile_$1_plus_real" ;; esac; }
ckpt_of() {  # <arm> <seed>
  local r; r=$(repo_of "$1")
  if [ "$1" = ctrl ] && [ "$2" = 1000 ]; then
    echo "$HOME/datastorage/nanowm_pusht_aug/value_model/sailor_expand/finetune/train/pusht_imgonly_d8_s1000/checkpoints/020000/pretrained_model"
  else
    printf '%s/%s_d8_s%s/checkpoints/020000/pretrained_model' "$O/train" "${r#local/}" "$2"
  fi
}
ft_unit() {  # <arm> <seed> <gpu>
  local ck run; ck=$(ckpt_of "$1" "$2"); run=${ck%/checkpoints/*}
  [ -d "$ck" ] && { echo "skip $1 $2"; return 0; }
  rm -rf "$run"
  CUDA_VISIBLE_DEVICES=$3 OUT_ROOT=$O/train SAVE_FREQ=20000 WANDB_ENABLE=${WANDB_ENABLE:-true} \
    bash $P/scripts/finetune_cl.sh "repo=$(repo_of "$1")" "$2" 20000 8 || return 1
  [ -d "$ck" ]
}
ev_unit() {  # <arm> <seed> <gpu>
  local out=$O/eval_seeds/$1_s$2
  [ -f "$out/success_steps.npz" ] && { echo "skip $1 $2"; return 0; }
  rm -rf "$out"; mkdir -p "$out" || return 1
  CUDA_VISIBLE_DEVICES=$3 $PY $P/eval/eval_success_steps.py --policy.path="$(ckpt_of "$1" "$2")" --policy.device=cuda \
    --policy.use_amp=false --policy.n_action_steps=10 --env.type=pusht --eval.n_episodes=2000 --eval.batch_size=50 \
    --eval.use_async_envs=false --seed=100000 --output_dir="$out"
}
in_pairs() {  # <unit fn> <name>: two units at a time, one per GPU
  local fn=$1 name=$2 i fail=0 p0 p1
  for ((i = 0; i < ${#UNITS[@]}; i += 2)); do
    read -r a0 s0 <<< "${UNITS[$i]}"
    ( "$fn" "$a0" "$s0" "${GPU_LIST[0]}" > "$O/logs/${name}_${a0}_s${s0}.log" 2>&1 ) & p0=$!
    p1=""
    if (( i + 1 < ${#UNITS[@]} )); then
      read -r a1 s1 <<< "${UNITS[$((i + 1))]}"
      ( "$fn" "$a1" "$s1" "${GPU_LIST[1 % ${#GPU_LIST[@]}]}" > "$O/logs/${name}_${a1}_s${s1}.log" 2>&1 ) & p1=$!
    fi
    wait "$p0" || { status "FAIL $name $a0 $s0"; fail=1; }
    if [ -n "$p1" ]; then wait "$p1" || { status "FAIL $name $a1 $s1"; fail=1; }; fi
    [ $fail = 0 ] || return 1
  done
}
st_finetune() { in_pairs ft_unit ft; }
st_eval() {
  mkdir -p "$O/eval_seeds"
  local arm
  for arm in W R ctrl; do [ -e "$O/eval_seeds/${arm}_s1000" ] || ln -s "$O/eval/$arm" "$O/eval_seeds/${arm}_s1000"; done
  in_pairs ev_unit ev
}
st_pooled() { $PY $P/critic/lastmile_pooled.py "$O/eval_seeds" --arms W R ctrl --seeds 1000 1001 --pairs W:R W:ctrl R:ctrl; }

status "START seed replication gpus=${GPU_LIST[*]}"
for s in finetune eval pooled; do
  [ -f "$O/.done_seeds/$s" ] && { status "skip $s (done)"; continue; }
  status "begin $s"
  if "st_$s"; then touch "$O/.done_seeds/$s"; status "done $s"
  else status "FAIL $s; re-run to resume"; FINISHED=1; exit 1; fi
done
FINISHED=1
status "ALL DONE"
