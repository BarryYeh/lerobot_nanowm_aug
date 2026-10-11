#!/usr/bin/env bash
# Full-trajectory upper-bound test: arm F (replicated branch replayed in the sim, then base policy to success) vs
# arm S (base policy's own successes from the same anchors), each as full episodes added to the real data.
# Criterion: <out_dir>/criteria.md, addendum 2 (fixed before F / S were generated).
#
#   bash run_traj_ft.sh tmux      detached (survives ssh loss); GPUS="6 7" by default
#   bash run_traj_ft.sh           in this shell
# Resumable: finished steps leave <out_dir>/.done_traj/<step>; progress in <out_dir>/status_traj.txt.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
CFG=configs/sailor_expand.yaml
P=examples/nanowm_pusht_aug
PY="uv run --extra training python"
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
y() { uv run python $P/attribution/config_io.py "$CFG" "$1"; }
OUT=$(y out_dir)

if [ "${1:-}" = "tmux" ]; then
  tmux has-session -t sailor_traj 2>/dev/null && { echo "tmux session sailor_traj already running"; exit 1; }
  tmux new-session -d -s sailor_traj "GPUS='${GPU_LIST[*]}' WANDB_ENABLE='${WANDB_ENABLE:-true}' bash '$SELF' 2>&1 | tee -a '$OUT/run_traj.log'"
  echo "started tmux session sailor_traj; log $OUT/run_traj.log, status $OUT/status_traj.txt"
  exit 0
fi

ST=$OUT/status_traj.txt
mkdir -p "$OUT/.done_traj" "$OUT/logs"
status() { local l; l="$(date '+%F %T') $*"; echo "$l" >> "$ST"; echo "$l" 2>/dev/null || true; }
FINISHED=0
trap '[ "$FINISHED" = 1 ] || status "FAIL traj driver exited unexpectedly (code $?); re-run to resume"' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM

ARMS=(traj_f traj_s)
POOLS=(local/pusht_traj_f local/pusht_traj_s)
FT_SEED=$(y expand.finetune.seed); FT_STEPS=$(y expand.finetune.steps); FT_DROP=$(y expand.finetune.drop)
FT_ROOT=$OUT/finetune/train
EV=$OUT/finetune/eval
ckpt_of() { printf '%s/%s_plus_real_d%s_s%s/checkpoints/%06d/pretrained_model' "$FT_ROOT" "${1#local/}" "$FT_DROP" "$FT_SEED" "$FT_STEPS"; }

step_gen() { CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} $PY $C/build_traj_datasets.py --config "$CFG"; }
C=$P/critic
step_build() {
  $PY - "$(y expand.finetune.real_repo)" "${POOLS[@]}" <<'EOF'
import json, sys
sys.path.insert(0, "examples/nanowm_pusht_aug/attribution")
from build_selected_datasets import build_real_plus_pairs
from lerobot.utils.constants import HF_LEROBOT_HOME
real, pools = sys.argv[1], sys.argv[2:]
stats = json.loads((HF_LEROBOT_HOME / real / "meta/stats.json").read_text())
for pool in pools:
    n_pool = json.loads((HF_LEROBOT_HOME / pool / "meta/info.json").read_text())["total_episodes"]
    n = build_real_plus_pairs(real, pool, list(range(n_pool)), f"{pool}_plus_real", stats)
    print(pool, n_pool, "->", n)
EOF
}
ft_unit() {
  local pool=${POOLS[$1]} ck run; ck=$(ckpt_of "$pool"); run=${ck%/checkpoints/*}
  [ -d "$ck" ] && { echo "skip ${ARMS[$1]}"; return 0; }
  rm -rf "$run"
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$(( $1 % ${#GPU_LIST[@]} ))]} OUT_ROOT=$FT_ROOT SAVE_FREQ=$FT_STEPS WANDB_ENABLE=${WANDB_ENABLE:-true} \
    bash $P/scripts/finetune_cl.sh "repo=${pool}_plus_real" "$FT_SEED" "$FT_STEPS" "$FT_DROP" || return 1
  [ -d "$ck" ]
}
ev_unit() {
  local out=$EV/${ARMS[$1]}
  [ -f "$out/success_steps.npz" ] && { echo "skip ${ARMS[$1]}"; return 0; }
  rm -rf "$out"; mkdir -p "$out" || return 1
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$(( $1 % ${#GPU_LIST[@]} ))]} $PY $P/eval/eval_success_steps.py \
    --policy.path="$(ckpt_of "${POOLS[$1]}")" --policy.device=cuda --policy.use_amp=false \
    --policy.n_action_steps="$(y expand.eval.n_action_steps)" --env.type=pusht \
    --eval.n_episodes="$(y expand.eval.n_episodes)" --eval.batch_size=50 --eval.use_async_envs=false \
    --seed="$(y expand.eval.seed)" --output_dir="$out"
}
par2() {
  local fail=0 p0 p1
  ( "$1" 0 > "$OUT/logs/traj_$2_0.log" 2>&1 ) & p0=$!
  ( "$1" 1 > "$OUT/logs/traj_$2_1.log" 2>&1 ) & p1=$!
  wait "$p0" || { status "FAIL $2 ${ARMS[0]} -> $OUT/logs/traj_$2_0.log"; fail=1; }
  wait "$p1" || { status "FAIL $2 ${ARMS[1]} -> $OUT/logs/traj_$2_1.log"; fail=1; }
  return $fail
}
step_finetune() { par2 ft_unit finetune; }
step_eval() { par2 ev_unit eval; }
step_compare() {
  local D=$OUT/finetune/eval_traj a
  mkdir -p "$D"
  for a in base ctrl "${ARMS[@]}"; do [ -e "$D/$a" ] || ln -s "$(readlink -f "$EV/$a")" "$D/$a"; done
  $PY $P/attribution/finetune_compare.py "$D" --arms base ctrl "${ARMS[@]}" \
    --pairs traj_f:ctrl traj_s:ctrl traj_f:traj_s traj_f:base traj_s:base
}
step_anchor() {
  CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} $PY $C/diag_anchor_rollouts.py --config "$CFG" \
    --arms base ctrl traj_f traj_s branch --video-arms traj_f traj_s ctrl --out "$OUT/diag_anchor_traj"
}

status "START traj test gpus=${GPU_LIST[*]}"
for s in gen build finetune eval compare anchor; do
  [ -f "$OUT/.done_traj/$s" ] && { status "skip $s (done)"; continue; }
  status "begin $s"
  if "step_$s"; then touch "$OUT/.done_traj/$s"; status "done $s"
  else status "FAIL $s; re-run to resume"; FINISHED=1; exit 1; fi
done
FINISHED=1
status "ALL DONE"
