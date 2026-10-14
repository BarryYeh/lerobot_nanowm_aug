#!/usr/bin/env bash
# Method 1, imagined-frame harm test (criterion: ~/datastorage/nanowm_pusht_aug/lastmile/method1/criteria.md).
# Prerequisite (done by hand, see the criterion): method1_prep.py pick -> realize_branches.py -> method1_prep.py label.
# Stages: pack (img_self / real_self pairs, last-mile anchor layout) -> build (real + pairs) -> finetune (2 arms x seeds
# 1000/1001) -> eval (2000 paired episodes) -> pooled (non-inferiority, margin 2 pp; ctrl = the last-mile ctrl runs)
# -> multiseed (OPTIONAL offline WM-seed spread; runs scripts/run_method1_multiseed.sh if it exists; never fatal).
#   bash run_method1.sh tmux    detached (survives ssh loss); GPUS="6 7" by default
# Resumable: finished stages leave <out>/.done/<stage>; progress in <out>/status.txt.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
P=examples/nanowm_pusht_aug
PY="uv run --extra training python"
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
LM=$HOME/datastorage/nanowm_pusht_aug/lastmile
O=$LM/method1

if [ "${1:-}" = "tmux" ]; then
  tmux has-session -t method1 2>/dev/null && { echo "tmux session method1 already running"; exit 1; }
  tmux new-session -d -s method1 "GPUS='${GPU_LIST[*]}' WANDB_ENABLE='${WANDB_ENABLE:-true}' bash '$SELF' 2>&1 | tee -a '$O/run.log'"
  echo "started tmux session method1; log $O/run.log, status $O/status.txt"
  exit 0
fi

mkdir -p "$O/.done" "$O/logs"
status() { local l; l="$(date '+%F %T') $*"; echo "$l" >> "$O/status.txt"; echo "$l" 2>/dev/null || true; }
FINISHED=0
trap '[ "$FINISHED" = 1 ] || status "FAIL driver exited unexpectedly (code $?); re-run to resume"' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM

ARMS=(img_self real_self)
UNITS=("img_self 1000" "real_self 1000" "img_self 1001" "real_self 1001")
repo_of() { echo "local/pusht_m1_$1_plus_real"; }
ckpt_of() { printf '%s/pusht_m1_%s_plus_real_d8_s%s/checkpoints/020000/pretrained_model' "$O/train" "$1" "$2"; }

st_pack() {
  local arm n
  for arm in "${ARMS[@]}"; do
    $PY $P/datagen/pack_closed_loop.py --accepted "$O/pairs_$arm" --repo-id "local/pusht_m1_$arm" \
      --ep-len 9 --pair-cov 0.0 --include-anchor --overwrite || return 1
    n=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['pairs']; assert all(p['j']==-1 for p in d); print(len(d))" \
      "$HOME/.cache/huggingface/lerobot/local/pusht_m1_$arm/meta/pair_index.json") || return 1
    [ "$n" = 121 ] || { echo "$arm packed $n pairs, expected 121"; return 1; }
  done
}
st_build() {
  $PY - "local/pusht_imgonly" "local/pusht_m1_img_self" "local/pusht_m1_real_self" <<'EOF'
import json, sys
sys.path.insert(0, "examples/nanowm_pusht_aug/attribution")
from build_selected_datasets import build_real_plus_pairs
from lerobot.utils.constants import HF_LEROBOT_HOME
real, pools = sys.argv[1], sys.argv[2:]
stats = json.loads((HF_LEROBOT_HOME / real / "meta/stats.json").read_text())
n = {}
for pool in pools:
    n_pool = json.loads((HF_LEROBOT_HOME / pool / "meta/info.json").read_text())["total_episodes"]
    n[pool] = (n_pool, build_real_plus_pairs(real, pool, list(range(n_pool)), f"{pool}_plus_real", stats))
    print(pool, n[pool])
assert {v[0] for v in n.values()} == {121}, n
EOF
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
  local out=$O/eval/$1_s$2
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
  mkdir -p "$O/eval"
  [ -e "$O/eval/ctrl_s1000" ] || ln -s "$(readlink -f "$LM/eval/ctrl")" "$O/eval/ctrl_s1000"
  [ -e "$O/eval/ctrl_s1001" ] || ln -s "$(readlink -f "$LM/eval_seeds/ctrl_s1001")" "$O/eval/ctrl_s1001"
  in_pairs ev_unit ev
}
st_pooled() {
  $PY $P/critic/lastmile_pooled.py "$O/eval" --arms img_self real_self ctrl --seeds 1000 1001 \
    --pairs img_self:real_self img_self:ctrl real_self:ctrl --margin 0.02
}
st_multiseed() {  # optional offline measurement; a failure is logged, never blocks the main result
  local s=$P/scripts/run_method1_multiseed.sh
  [ -f "$s" ] || { status "multiseed: $s not found, skipped"; return 0; }
  GPUS="${GPU_LIST[*]}" bash "$s" > "$O/logs/multiseed.log" 2>&1 || status "multiseed FAILED (non-fatal), see logs/multiseed.log"
  return 0
}

status "START method1 gpus=${GPU_LIST[*]}"
for s in pack build finetune eval pooled; do
  [ -f "$O/.done/$s" ] && { status "skip $s (done)"; continue; }
  status "begin $s"
  if "st_$s"; then touch "$O/.done/$s"; status "done $s"
  else status "FAIL $s; re-run to resume"; FINISHED=1; exit 1; fi
done
status "MAIN DONE"
if [ ! -f "$O/.done/multiseed" ]; then status "begin multiseed"; st_multiseed; touch "$O/.done/multiseed"; status "done multiseed"; fi
FINISHED=1
status "ALL DONE"
