#!/usr/bin/env bash
# Last-mile experiment (criterion: ~/datastorage/nanowm_pusht_aug/lastmile/criteria.md, fixed before any data).
# Stuck late states from base-policy deployment -> 12 candidate chunks per start (k 4 x sigma {0,0.1,0.25}) imagined
# for 2 NanoWM rounds -> W = best imagined coverage, R = random -> gate 1 (>= 100 accepted) and gate 2 (simulator
# replay, diagnostic only: W true success - R >= 10 pp) -> real anchor frame + executed chunk pairs, ~5% of samples
# -> fine-tune W and R -> 2000 paired episodes for W, R, ctrl.
#
#   bash run_lastmile.sh tmux [--smoke]      detached (survives ssh loss); GPUS="6 7" by default
#   bash run_lastmile.sh [--smoke]           in this shell
# Resumable: finished stages leave <out>/.done/<stage>; progress in <out>/status.txt. --smoke uses <out>/smoke, tiny
# sizes, no gates, and stops after build.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
P=examples/nanowm_pusht_aug
C=$P/critic
PY="uv run --extra training python"
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
SMOKE=""; for a in "$@"; do [ "$a" = "--smoke" ] && SMOKE=1; done
ROOT=$HOME/datastorage/nanowm_pusht_aug/lastmile
O=$ROOT${SMOKE:+/smoke}

if [ "${1:-}" = "tmux" ]; then
  S=lastmile${SMOKE:+_smoke}
  tmux has-session -t "$S" 2>/dev/null && { echo "tmux session $S already running"; exit 1; }
  mkdir -p "$O"
  tmux new-session -d -s "$S" "GPUS='${GPU_LIST[*]}' WANDB_ENABLE='${WANDB_ENABLE:-true}' bash '$SELF' ${SMOKE:+--smoke} 2>&1 | tee -a '$O/run.log'"
  echo "started tmux session $S; log $O/run.log, status $O/status.txt"
  exit 0
fi

mkdir -p "$O/.done" "$O/logs"
status() { local l; l="$(date '+%F %T') $*"; echo "$l" >> "$O/status.txt"; echo "$l" 2>/dev/null || true; }
FINISHED=0
trap '[ "$FINISHED" = 1 ] || status "FAIL driver exited unexpectedly (code $?); re-run to resume"' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM

BASE_CKPT=outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model
CTRL_CKPT=$HOME/datastorage/nanowm_pusht_aug/value_model/sailor_expand/finetune/train/pusht_imgonly_d8_s1000/checkpoints/020000/pretrained_model
if [ -n "$SMOKE" ]; then N=6; K=1; MIN_ACC=0; MIN_GAP=-1; TARGET=12; else N=400; K=4; MIN_ACC=100; MIN_GAP=0.10; TARGET=1200; fi
SIGMAS=(0.0 0.1 0.25)
NSS=$(uv run python $P/attribution/config_io.py configs/nanowm_pusht_common.yaml nanowm.num_sampling_steps)
WMB=$(uv run python $P/attribution/config_io.py configs/nanowm_pusht_common.yaml nanowm.wm_batch)
SUF=${SMOKE:+_smoke}

st_collect() {
  [ -f "$O/seeds_late.npz" ] && return 0
  CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} $PY $C/collect_late_states.py --ckpt "$BASE_CKPT" --out "$O/seeds_late.npz" --n "$N"
}
imagine_sigma() {  # <sigma> <gpu>
  local SIG=$1 RUN=$O/imag/sigma$1 LEADARG=() want=$(( N * K )) have
  have=$(find "$RUN/branches" -name 'seed*_cand*.npz' 2>/dev/null | wc -l)
  [ "$have" -eq "$want" ] && { echo "sigma $SIG complete"; return 0; }
  [ "$SIG" != "0.0" ] && LEADARG=(--lead-in "$O/imag/sigma0.0/work/lead_in.npz")
  rm -rf "$RUN/branches"; mkdir -p "$RUN/branches" || return 1
  CUDA_VISIBLE_DEVICES=$2 $PY $P/eval/closed_loop_rollout.py --ckpt "$BASE_CKPT" --seeds "$O/seeds_late.npz" \
    --n-seeds "$N" --k "$K" --h-max 2 --inject-sigma "$SIG" --accept-cov 0 --accept-mode terminal --save-latents \
    --replan-every-rounds 2 --noise-hold 1 --max-z 3.0 --lead-na 10 --torch-seed 950000 --noise-seed 950003 \
    ${LEADARG[@]+"${LEADARG[@]}"} --num-sampling-steps "$NSS" --wm-batch "$WMB" \
    --work "$RUN/work" --out "$RUN/branches" || return 1
  have=$(find "$RUN/branches" -name 'seed*_cand*.npz' | wc -l)
  [ "$have" -eq "$want" ] || { echo "sigma $SIG wrote $have of $want"; return 1; }
}
st_imagine() {
  imagine_sigma 0.0 "${GPU_LIST[0]}" > "$O/logs/imagine_0.0.log" 2>&1 || { status "FAIL imagine sigma 0.0"; return 1; }
  local fail=0 p1 p2
  ( imagine_sigma 0.1 "${GPU_LIST[0]}" > "$O/logs/imagine_0.1.log" 2>&1 ) & p1=$!
  ( imagine_sigma 0.25 "${GPU_LIST[1 % ${#GPU_LIST[@]}]}" > "$O/logs/imagine_0.25.log" 2>&1 ) & p2=$!
  wait "$p1" || { status "FAIL imagine sigma 0.1"; fail=1; }
  wait "$p2" || { status "FAIL imagine sigma 0.25"; fail=1; }
  return $fail
}
st_select() { $PY $C/select_lastmile.py --root "$O/imag" --out "$O/sel" --min-accepted "$MIN_ACC" ${SMOKE:+--force-min 2}; }
st_simcheck() { $PY $C/lastmile_simcheck.py --root "$O/imag" --sel "$O/sel" --min-gap "$MIN_GAP"; }
st_pack() {
  local n dup arm f k
  n=$(ls "$O/sel/W" | wc -l)
  dup=$(( (TARGET + n - 1) / n ))
  echo "accepted $n -> duplicate x$dup"
  for arm in W R; do
    rm -rf "$O/sel/${arm}_dup"; mkdir -p "$O/sel/${arm}_dup"
    for f in "$O/sel/$arm"/*.npz; do
      for ((k = 0; k < dup; k++)); do ln -s "$(readlink -f "$f")" "$O/sel/${arm}_dup/dup${k}_$(basename "$f")"; done
    done
    $PY $P/datagen/pack_closed_loop.py --accepted "$O/sel/${arm}_dup" --repo-id "local/pusht_lastmile_${arm}$SUF" \
      --ep-len 9 --pair-cov 0.0 --include-anchor --overwrite || return 1
  done
}
st_build() {
  $PY - "local/pusht_imgonly" "local/pusht_lastmile_W$SUF" "local/pusht_lastmile_R$SUF" <<'EOF'
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
assert len({v[0] for v in n.values()}) == 1, n
EOF
}
ckpt_of() { printf '%s/pusht_lastmile_%s_plus_real_d8_s1000/checkpoints/020000/pretrained_model' "$O/train" "$1"; }
ft_unit() {
  local arm=$1 ck run; ck=$(ckpt_of "$arm"); run=${ck%/checkpoints/*}
  [ -d "$ck" ] && { echo "skip $arm"; return 0; }
  rm -rf "$run"
  CUDA_VISIBLE_DEVICES=$2 OUT_ROOT=$O/train SAVE_FREQ=20000 WANDB_ENABLE=${WANDB_ENABLE:-true} \
    bash $P/scripts/finetune_cl.sh "repo=local/pusht_lastmile_${arm}_plus_real" 1000 20000 8 || return 1
  [ -d "$ck" ]
}
st_finetune() {
  [ -n "$SMOKE" ] && return 0
  local fail=0 p1 p2
  ( ft_unit W "${GPU_LIST[0]}" > "$O/logs/ft_W.log" 2>&1 ) & p1=$!
  ( ft_unit R "${GPU_LIST[1 % ${#GPU_LIST[@]}]}" > "$O/logs/ft_R.log" 2>&1 ) & p2=$!
  wait "$p1" || { status "FAIL finetune W"; fail=1; }
  wait "$p2" || { status "FAIL finetune R"; fail=1; }
  return $fail
}
ev_unit() {  # <arm> <ckpt> <gpu>
  local out=$O/eval/$1
  [ -f "$out/success_steps.npz" ] && { echo "skip $1"; return 0; }
  rm -rf "$out"; mkdir -p "$out" || return 1
  CUDA_VISIBLE_DEVICES=$3 $PY $P/eval/eval_success_steps.py --policy.path="$2" --policy.device=cuda \
    --policy.use_amp=false --policy.n_action_steps=10 --env.type=pusht --eval.n_episodes=2000 --eval.batch_size=50 \
    --eval.use_async_envs=false --seed=100000 --output_dir="$out"
}
st_eval() {
  [ -n "$SMOKE" ] && return 0
  local fail=0 p1 p2
  ( ev_unit W "$(ckpt_of W)" "${GPU_LIST[0]}" > "$O/logs/ev_W.log" 2>&1 ) & p1=$!
  ( ev_unit R "$(ckpt_of R)" "${GPU_LIST[1 % ${#GPU_LIST[@]}]}" > "$O/logs/ev_R.log" 2>&1 ) & p2=$!
  wait "$p1" || { status "FAIL eval W"; fail=1; }
  wait "$p2" || { status "FAIL eval R"; fail=1; }
  [ $fail = 0 ] || return 1
  ev_unit ctrl "$CTRL_CKPT" "${GPU_LIST[0]}" > "$O/logs/ev_ctrl.log" 2>&1 || { status "FAIL eval ctrl"; return 1; }
}
st_compare() {
  [ -n "$SMOKE" ] && return 0
  $PY $P/attribution/finetune_compare.py "$O/eval" --arms W R ctrl --pairs W:R W:ctrl R:ctrl || return 1
  $PY - "$O/eval" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
ev = Path(sys.argv[1]); out = {}
for arm in ("W", "R", "ctrl"):
    m = json.loads((ev / arm / "eval_info.json").read_text())["per_task"][0]["metrics"]
    s, mr = np.asarray(m["successes"], bool), np.asarray(m["max_rewards"], float)
    # gym_pusht reward = clip(coverage / 0.95, 0, 1): max coverage >= 0.9 <=> max reward >= 0.9 / 0.95
    out[arm] = {"success": float(s.mean()), "failures": int((~s).sum()), "failures_reaching_cov_0.9": int(((~s) & (mr >= 0.9 / 0.95)).sum())}
(ev / "secondary.json").write_text(json.dumps(out, indent=1)); print(out)
EOF
}

status "START lastmile ${SMOKE:+(smoke)} gpus=${GPU_LIST[*]}"
for s in collect imagine select simcheck pack build finetune eval compare; do
  [ -f "$O/.done/$s" ] && { status "skip $s (done)"; continue; }
  status "begin $s"
  if "st_$s"; then touch "$O/.done/$s"; status "done $s"
  else status "FAIL $s (gates stop here by design; see logs / run.log)"; FINISHED=1; exit 1; fi
done
FINISHED=1
status "ALL DONE"
