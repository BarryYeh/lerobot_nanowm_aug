#!/usr/bin/env bash
# imagined_replay step 4b-5 (settings: ~/datastorage/nanowm_pusht_aug/imagined_replay/criteria.md).
# Prerequisite: pack_pairs.py select -> pack -> build (local/pusht_ir_{real,img}_plus_real).
# Stages: finetune (real / img x seeds 1000/1001, 20k steps, drop 8) -> eval (2000 paired episodes, seed 100000,
# na 10) -> pooled (lastmile_pooled.py; ctrl = the last-mile ctrl evaluations, same recipe and episodes).
#   bash run_imagined_replay.sh tmux    detached (survives ssh loss); GPUS="6 7" by default
# Resumable: finished stages leave <out>/.done/<stage>; progress in <out>/status.txt.
# Other runs of the same design (defaults = the main Real / Imagined run): IR_DIR (dir under imagined_replay/),
# IR_TAG (repo prefix: local/pusht_<tag>_<arm>_plus_real; also names the tmux session), IR_ARMS (arms to train),
# IR_NEP (eval episodes per seed), IR_REUSE (last-mile arms whose evaluations are reused: <arm>_s1000 =
# lastmile/eval/<arm>, <arm>_s1001 = lastmile/eval_seeds/<arm>_s1001; with IR_NEP < 2000 they are cut to their first
# IR_NEP episodes, which have the same env seeds 100000+i), IR_PAIRS (pooled comparisons). Img-W on the last-mile W set:
#   IR_DIR=lastmile_w/finetune IR_TAG=irw IR_ARMS=imgw IR_NEP=1000 IR_REUSE="W ctrl" \
#   IR_PAIRS="imgw:W imgw:ctrl W:ctrl" bash run_imagined_replay.sh tmux
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
RES=research/nanowm_aug  # the one PYTHONPATH root
PY="uv run --extra training python"
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
LM=$HOME/datastorage/nanowm_pusht_aug/lastmile
IR_DIR=${IR_DIR:-finetune}
O=$HOME/datastorage/nanowm_pusht_aug/imagined_replay/$IR_DIR
TAG=${IR_TAG:-ir}
read -r -a ARMS <<< "${IR_ARMS:-real img}"
NEP=${IR_NEP:-2000}
read -r -a REUSE <<< "${IR_REUSE:-ctrl}"
read -r -a PAIRS <<< "${IR_PAIRS:-img:real img:ctrl real:ctrl}"
SESSION=imagined_replay_${TAG}_ft

if [ "${1:-}" = "tmux" ]; then
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session $SESSION already running"; exit 1; }
  mkdir -p "$O"
  tmux new-session -d -s "$SESSION" "GPUS='${GPU_LIST[*]}' WANDB_ENABLE='${WANDB_ENABLE:-true}' IR_DIR='$IR_DIR' \
IR_TAG='$TAG' IR_ARMS='${ARMS[*]}' IR_NEP='$NEP' IR_REUSE='${REUSE[*]}' IR_PAIRS='${PAIRS[*]}' \
bash '$SELF' 2>&1 | tee -a '$O/run.log'"
  echo "started tmux session $SESSION; log $O/run.log, status $O/status.txt"
  exit 0
fi

mkdir -p "$O/.done" "$O/logs"
status() { local l; l="$(date '+%F %T') $*"; echo "$l" >> "$O/status.txt"; echo "$l" 2>/dev/null || true; }
FINISHED=0
trap '[ "$FINISHED" = 1 ] || status "FAIL driver exited unexpectedly (code $?); re-run to resume"' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM

UNITS=()
for s in 1000 1001; do for a in "${ARMS[@]}"; do UNITS+=("$a $s"); done; done
repo_of() { echo "local/pusht_${TAG}_$1_plus_real"; }
ckpt_of() { printf '%s/pusht_%s_%s_plus_real_d8_s%s/checkpoints/020000/pretrained_model' "$O/train" "$TAG" "$1" "$2"; }

ft_unit() {  # <arm> <seed> <gpu>
  local ck run; ck=$(ckpt_of "$1" "$2"); run=${ck%/checkpoints/*}
  [ -d "$ck" ] && { echo "skip $1 $2"; return 0; }
  rm -rf "$run"
  CUDA_VISIBLE_DEVICES=$3 OUT_ROOT=$O/train SAVE_FREQ=20000 WANDB_ENABLE=${WANDB_ENABLE:-true} \
    bash $RES/proj/pusht/scripts/finetune_cl.sh "repo=$(repo_of "$1")" "$2" 20000 8 || return 1
  [ -d "$ck" ]
}
ev_unit() {  # <arm> <seed> <gpu>
  local out=$O/eval/$1_s$2
  [ -f "$out/success_steps.npz" ] && { echo "skip $1 $2"; return 0; }
  rm -rf "$out"; mkdir -p "$out" || return 1
  CUDA_VISIBLE_DEVICES=$3 $PY $RES/nwaug/eval/eval_success_steps.py --policy.path="$(ckpt_of "$1" "$2")" --policy.device=cuda \
    --policy.use_amp=false --policy.n_action_steps=10 --env.type=pusht --eval.n_episodes="$NEP" --eval.batch_size=50 \
    --eval.use_async_envs=false --seed=100000 --output_dir="$out"
}
reuse_eval() {  # <last-mile eval dir> <dest>: link it, or keep only its first NEP episodes
  local src dst=$2; src=$(readlink -f "$1")
  [ -f "$src/eval_info.json" ] || { echo "missing $src/eval_info.json"; return 1; }
  [ -e "$dst" ] && return 0
  if [ "$NEP" = 2000 ]; then ln -s "$src" "$dst"; return 0; fi
  mkdir -p "$dst" && $PY $RES/proj/pusht/imagined_replay/truncate_eval.py "$src" "$dst" "$NEP"
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
  mkdir -p "$O/eval"
  local a
  for a in "${REUSE[@]}"; do
    reuse_eval "$LM/eval/$a" "$O/eval/${a}_s1000" || return 1
    reuse_eval "$LM/eval_seeds/${a}_s1001" "$O/eval/${a}_s1001" || return 1
  done
  in_pairs ev_unit ev
}
st_pooled() {
  $PY $RES/proj/pusht/heuristic/lastmile_pooled.py "$O/eval" --arms "${ARMS[@]}" "${REUSE[@]}" --seeds 1000 1001 \
    --pairs "${PAIRS[@]}"
}

status "START imagined_replay $IR_DIR gpus=${GPU_LIST[*]} arms=${ARMS[*]} reuse=${REUSE[*]} episodes=$NEP"
for s in finetune eval pooled; do
  [ -f "$O/.done/$s" ] && { status "skip $s (done)"; continue; }
  status "begin $s"
  if "st_$s"; then touch "$O/.done/$s"; status "done $s"
  else status "FAIL $s; re-run to resume"; FINISHED=1; exit 1; fi
done
FINISHED=1
status "ALL DONE"
