#!/usr/bin/env bash
# SAILOR expand dose test: the replicated branches (selected/ and realized/ of run_sailor_expand.sh) duplicated DUP
# times in the pair pool, so the same pairs carry ~DUP x the weight in fine-tuning. No training code changes: the
# packer sees DUP symlinks per branch (dup<k>_<name>.npz). Arms img_x<DUP> (NanoWM frames) and real_x<DUP> (sim
# frames), same fine-tune seed / steps / drop and the same eval as the main run; ctrl and base are reused.
# Criterion: <out_dir>/criteria.md (addendum), fixed before these arms were trained.
#
#   bash run_sailor_expand_dose.sh tmux        detached (survives ssh loss); DUP=10 GPUS="6 7" by default
#   bash run_sailor_expand_dose.sh             in this shell
# Resumable like run_sailor_expand.sh: finished steps leave <out_dir>/.done_dose/<step>; progress in status_dose.txt.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
CFG=configs/sailor_expand.yaml
P=examples/nanowm_pusht_aug
PY="uv run --extra training python"
DUP=${DUP:-10}
X=x$DUP
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
y() { uv run python $P/attribution/config_io.py "$CFG" "$1"; }
OUT=$(y out_dir)

if [ "${1:-}" = "tmux" ]; then
  tmux has-session -t sailor_dose 2>/dev/null && { echo "tmux session sailor_dose already running"; exit 1; }
  tmux new-session -d -s sailor_dose "DUP=$DUP GPUS='${GPU_LIST[*]}' WANDB_ENABLE='${WANDB_ENABLE:-false}' bash '$SELF' 2>&1 | tee -a '$OUT/run_dose.log'"
  echo "started tmux session sailor_dose; log $OUT/run_dose.log, status $OUT/status_dose.txt"
  exit 0
fi

ST=$OUT/status_dose.txt
mkdir -p "$OUT/.done_dose" "$OUT/logs"
status() { local l; l="$(date '+%F %T') $*"; echo "$l" >> "$ST"; echo "$l" 2>/dev/null || true; }
FINISHED=0
trap '[ "$FINISHED" = 1 ] || status "FAIL dose driver exited unexpectedly (code $?); re-run to resume"' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM

IMG_POOL=$(y expand.finetune.img_repo)_$X
REAL_POOL=$(y expand.finetune.real_frames_repo)_$X
FT_SEED=$(y expand.finetune.seed); FT_STEPS=$(y expand.finetune.steps); FT_DROP=$(y expand.finetune.drop)
FT_ROOT=$OUT/finetune/train
EV=$OUT/finetune/eval
ARMS=("img_$X" "real_$X")
POOLS=("$IMG_POOL" "$REAL_POOL")

step_dup() {
  for src in selected realized; do
    local d=$OUT/${src}_$X
    [ -d "$d" ] && continue
    rm -rf "$d.partial"; mkdir -p "$d.partial"
    for f in "$OUT/$src"/*.npz; do
      for ((k = 0; k < DUP; k++)); do ln -s "$(readlink -f "$f")" "$d.partial/dup${k}_$(basename "$f")"; done
    done
    mv "$d.partial" "$d"
    echo "$d: $(ls "$d" | wc -l) files"
  done
}
step_pack() {
  local EL; EL=$(y expand.pack.ep_len)
  $PY $P/datagen/pack_closed_loop.py --accepted "$OUT/selected_$X" --repo-id "$IMG_POOL" --ep-len "$EL" \
    --pair-cov "$(y expand.pack.pair_cov)" --include-anchor --overwrite || return 1
  $PY $P/datagen/pack_closed_loop.py --accepted "$OUT/realized_$X" --repo-id "$REAL_POOL" --ep-len "$EL" \
    --pair-cov "$(y expand.pack.pair_cov)" --include-anchor --overwrite
}
step_build() {
  $PY - "$(y expand.finetune.real_repo)" "$IMG_POOL" "$REAL_POOL" "$OUT/datasets_$X.json" <<'EOF'
import json, sys
sys.path.insert(0, "examples/nanowm_pusht_aug/attribution")
from build_selected_datasets import build_real_plus_pairs
from lerobot.utils.constants import HF_LEROBOT_HOME
real, pools, out = sys.argv[1], sys.argv[2:4], sys.argv[4]
stats = json.loads((HF_LEROBOT_HOME / real / "meta/stats.json").read_text())
man = {}
for pool in pools:
    n_pool = json.loads((HF_LEROBOT_HOME / pool / "meta/info.json").read_text())["total_episodes"]
    n = build_real_plus_pairs(real, pool, list(range(n_pool)), f"{pool}_plus_real", stats)
    man[pool] = {"repo": f"{pool}_plus_real", "n_pairs": n_pool, "total_episodes": n}
    print(pool, man[pool])
assert len({m["n_pairs"] for m in man.values()}) == 1, man
open(out, "w").write(json.dumps(man, indent=1))
EOF
}
ckpt_of() { printf '%s/%s_plus_real_d%s_s%s/checkpoints/%06d/pretrained_model' "$FT_ROOT" "${1#local/}" "$FT_DROP" "$FT_SEED" "$FT_STEPS"; }
ft_unit() {  # <i>
  local pool=${POOLS[$1]} ck run; ck=$(ckpt_of "${POOLS[$1]}"); run=${ck%/checkpoints/*}
  [ -d "$ck" ] && { echo "skip ${ARMS[$1]} (checkpoint exists)"; return 0; }
  rm -rf "$run"
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$(( $1 % ${#GPU_LIST[@]} ))]} OUT_ROOT=$FT_ROOT SAVE_FREQ=$FT_STEPS WANDB_ENABLE=${WANDB_ENABLE:-false} \
    bash $P/scripts/finetune_cl.sh "repo=${pool}_plus_real" "$FT_SEED" "$FT_STEPS" "$FT_DROP" || return 1
  [ -d "$ck" ]
}
ev_unit() {
  local out=$EV/${ARMS[$1]}
  [ -f "$out/success_steps.npz" ] && { echo "skip ${ARMS[$1]} (evaluated)"; return 0; }
  rm -rf "$out"; mkdir -p "$out" || return 1
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$(( $1 % ${#GPU_LIST[@]} ))]} $PY $P/eval/eval_success_steps.py \
    --policy.path="$(ckpt_of "${POOLS[$1]}")" --policy.device=cuda --policy.use_amp=false \
    --policy.n_action_steps="$(y expand.eval.n_action_steps)" --env.type=pusht \
    --eval.n_episodes="$(y expand.eval.n_episodes)" --eval.batch_size=50 --eval.use_async_envs=false \
    --seed="$(y expand.eval.seed)" --output_dir="$out"
}
par2() {  # <unit fn> <name>: the two arms in parallel
  local fail=0 p0 p1
  ( "$1" 0 > "$OUT/logs/dose_$2_0.log" 2>&1 ) & p0=$!
  ( "$1" 1 > "$OUT/logs/dose_$2_1.log" 2>&1 ) & p1=$!
  wait "$p0" || { status "FAIL $2 ${ARMS[0]} -> $OUT/logs/dose_$2_0.log"; fail=1; }
  wait "$p1" || { status "FAIL $2 ${ARMS[1]} -> $OUT/logs/dose_$2_1.log"; fail=1; }
  return $fail
}
step_finetune() { par2 ft_unit finetune; }
step_eval() { par2 ev_unit eval; }
step_compare() {  # own dir: the main run's finetune/eval/comparison.md stays as it is
  local D=$OUT/finetune/eval_dose_$X a
  mkdir -p "$D"
  for a in base ctrl img real "${ARMS[@]}"; do [ -e "$D/$a" ] || ln -s "$(readlink -f "$EV/$a")" "$D/$a"; done
  $PY $P/attribution/finetune_compare.py "$D" --arms base ctrl img real "${ARMS[@]}" \
    --pairs "img_$X:ctrl" "real_$X:ctrl" "img_$X:img" "real_$X:real" "img_$X:real_$X"
}

status "START dose test DUP=$DUP gpus=${GPU_LIST[*]}"
for s in dup pack build finetune eval compare; do
  [ -f "$OUT/.done_dose/${s}_$X" ] && { status "skip $s (done)"; continue; }
  status "begin $s"
  if "step_$s"; then touch "$OUT/.done_dose/${s}_$X"; status "done $s"
  else status "FAIL $s; re-run to resume"; FINISHED=1; exit 1; fi
done
FINISHED=1
status "ALL DONE"
