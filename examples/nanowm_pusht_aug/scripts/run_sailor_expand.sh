#!/usr/bin/env bash
# SAILOR expand driver (configs/sailor_expand.yaml; plan ~/.claude/plans/1-anchor-stateless-possum.md).
#
#   bash run_sailor_expand.sh tmux [stage ...] [--smoke]   start the run DETACHED in tmux (survives ssh loss)
#   bash run_sailor_expand.sh [stage ...] [--smoke]        run in this shell (what the tmux session executes)
#
# Stages, in order (default `all`):
#   anchors   S1 candidate anchors (make_reset_anchors.py)        -> seeds_candidates.npz
#   screen    S2 base-policy screen, keep anchors with headroom    -> seeds_stage2.npz, screen.json
#   imagine   S3 NanoWM branches, sigma 0.0 first, then the others in parallel (replaying its lead-in)
#   verify    S4 discovery label passes (seed verify.seed + 2i)  -> stage2/labels_v<i>.npz
#   discover  S5 BH-FDR classes + per-anchor cap                  -> discovery.json, candidates.json, replicate/
#   replicate S5 fresh-seed label passes on the candidates        -> replicate/stage2/labels_rep_<i>.npz
#   finalize  S5 one-sided replication test                       -> selection.json, selected/
#   score_v   S6 value-model scores vs the 20-pass truth (V never selects)   -> v_eval.{json,md}
#   realize   S7 arm B: the selected branches with real sim frames           -> realized/
#   pack      S7 pack arms A (imagined) and B (real frames) into LeRobot pair datasets
#   build     S7 real + pairs datasets                            -> datasets.json
#   finetune  S7 arms img / real / ctrl, one seed each            (skipped with --smoke)
#   eval      S8 1000 paired episodes per arm with success steps  (skipped with --smoke)
#   compare   S8 paired success / step tables                     -> finetune/eval/comparison.md
#
# Resumable: a finished stage writes <base>/.done/<stage> and is skipped next time; inside a stage every unit
# (screen shard, sigma, label pass, fine-tune arm, eval arm) is skipped when its output already exists, and
# outputs are written under a temporary name and renamed when complete. Re-running the same command after a
# crash or an ssh drop therefore continues where it stopped. Progress: <base>/status.txt (one line per event).
# GPUs: GPUS="6 7" (default) -- parallel units are assigned round robin.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# NOTE: stages run inside `if st_<stage>` (errexit is off there), so every multi-command stage chains with || return 1
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
CFG=${CFG:-configs/sailor_expand.yaml}
P=examples/nanowm_pusht_aug
C=$P/critic
PY="uv run --extra training python"
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
NG=${#GPU_LIST[@]}

ALL="anchors screen imagine verify discover replicate finalize score_v realize pack build finetune eval compare"
SMOKE=""; ARGS=()
for a in "$@"; do if [ "$a" = "--smoke" ]; then SMOKE="--smoke"; else ARGS+=("$a"); fi; done

if [ "${ARGS[0]:-}" = "tmux" ]; then
  SESSION=sailor_expand${SMOKE:+_smoke}
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session $SESSION already running: tmux attach -t $SESSION"; exit 1; }
  rest=("${ARGS[@]:1}")
  OUTD=$(uv run python $P/attribution/config_io.py "$CFG" out_dir)${SMOKE:+/smoke}
  mkdir -p "$OUTD"
  tmux new-session -d -s "$SESSION" \
    "GPUS='${GPUS:-6 7}' CFG='$CFG' bash '$SELF' ${rest[*]} $SMOKE 2>&1 | tee -a '$OUTD/run.log'"
  echo "started tmux session $SESSION (tmux attach -t $SESSION); log $OUTD/run.log, status $OUTD/status.txt"
  exit 0
fi

y() { uv run python $P/attribution/config_io.py "$CFG" "$1"; }
ys() { if [ -n "$SMOKE" ]; then y "expand.smoke.$1"; else y "$2"; fi; }   # smoke key, normal key
OUT=$(y out_dir)
BASE=$OUT${SMOKE:+/smoke}
S2=$BASE/stage2
LOGS=$BASE/logs
mkdir -p "$BASE/.done" "$LOGS"
STAGES="${ARGS[*]:-all}"; [ "$STAGES" = "all" ] && STAGES=$ALL

status() {  # the file first: when tmux / ssh goes away, stdout (the tee to run.log) may already be gone
  local line; line="$(date '+%F %T') $*"
  echo "$line" >> "$BASE/status.txt"
  echo "$line" 2>/dev/null || true
}
gpu() { echo "${GPU_LIST[$(( $1 % NG ))]}"; }
# run_par <name> <cmd builder fn> <n units>: runs units in parallel, at most NPAR at a time; fails if any unit fails
run_par() {
  local name=$1 fn=$2 n=$3
  local npar=${NPAR:-$n} i pids=() fail=0
  for ((i = 0; i < n; i++)); do
    while [ "$(jobs -rp | wc -l)" -ge "$npar" ]; do sleep 2; done
    ( "$fn" "$i" > "$LOGS/${name}_$i.log" 2>&1 ) &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then status "FAIL $name unit $i -> $LOGS/${name}_$i.log"; fail=1; fi
  done
  return $fail
}

N_ANCH=$(ys n_anchors expand.n_anchors)
K=$(ys k stage2.k)
H_MAX=$(y stage2.h_max)
SIGMAS=$(y stage2.inject_sigma | tr -d '[],')

# ---------------------------------------------------------------- stage bodies
st_anchors() {
  [ -f "$BASE/seeds_candidates.npz" ] && return 0
  $PY $C/make_reset_anchors.py --config "$CFG" --section stage2.anchors --out "$BASE/seeds_candidates.npz" $SMOKE
}

SHARDS=$([ -n "$SMOKE" ] && echo 2 || y expand.screen.shards)
screen_unit() { CUDA_VISIBLE_DEVICES=$(gpu "$1") $PY $C/select_improvable_anchors.py --config "$CFG" $SMOKE screen --shard "$1" --n-shards "$SHARDS"; }
st_screen() {
  run_par screen screen_unit "$SHARDS" || return 1
  $PY $C/select_improvable_anchors.py --config "$CFG" $SMOKE merge --n-shards "$SHARDS"
}

imagine_sigma() {  # <sigma> <gpu>
  local SIG=$1 RUN=$S2/sigma$1 LEADARG=() want=$(( N_ANCH * K ))
  local have; have=$(find "$RUN/branches" -name 'seed*_cand*.npz' 2>/dev/null | wc -l)
  [ "$have" -eq "$want" ] && { echo "sigma $SIG complete ($have)"; return 0; }
  [ "$SIG" != "0.0" ] && LEADARG=(--lead-in "$S2/sigma0.0/work/lead_in.npz")
  rm -rf "$RUN/branches"; mkdir -p "$RUN/branches" || return 1
  CUDA_VISIBLE_DEVICES=$2 $PY $P/eval/closed_loop_rollout.py --ckpt "$(y "ckpt.$(y ckpt_key)")" \
    --seeds "$BASE/seeds_stage2.npz" --n-seeds "$N_ANCH" --k "$K" --h-max "$H_MAX" --inject-sigma "$SIG" \
    --accept-cov 0 --accept-mode terminal --save-latents \
    --replan-every-rounds "$(y stage2.replan_every_rounds)" --noise-hold "$(y stage2.noise_hold)" \
    --max-z "$(y stage2.max_z)" --lead-na "$(y stage2.lead_na)" \
    --torch-seed "$(y stage2.torch_seed)" --noise-seed "$(y stage2.noise_seed)" ${LEADARG[@]+"${LEADARG[@]}"} \
    --num-sampling-steps "$(y nanowm.num_sampling_steps)" --wm-batch "$(y nanowm.wm_batch)" \
    --work "$RUN/work" --out "$RUN/branches" || return 1
  have=$(find "$RUN/branches" -name 'seed*_cand*.npz' | wc -l)
  [ "$have" -eq "$want" ] || { echo "sigma $SIG wrote $have of $want branches"; return 1; }
}
OTHER_SIGMAS=()
for s in $SIGMAS; do [ "$s" != "0.0" ] && OTHER_SIGMAS+=("$s"); done
imagine_other_unit() { imagine_sigma "${OTHER_SIGMAS[$1]}" "$(gpu "$1")"; }
st_imagine() {
  [ "${SIGMAS%% *}" = "0.0" ] || { status "FAIL stage2.inject_sigma must start with 0.0"; return 1; }
  imagine_sigma 0.0 "$(gpu 0)" > "$LOGS/imagine_sigma0.0.log" 2>&1 || { status "FAIL imagine sigma 0.0 -> $LOGS/imagine_sigma0.0.log"; return 1; }
  run_par imagine_other imagine_other_unit "${#OTHER_SIGMAS[@]}"
}

# label pass unit: <cfg> <root> <smokeflag> <seed> <final name>
label_pass() {
  local cfg=$1 root=$2 sm=$3 seed=$4 name=$5 g=$6
  [ -f "$root/$name.npz" ] && { echo "skip $name (exists)"; return 0; }
  CUDA_VISIBLE_DEVICES=$g $PY $C/label_imagined_branches.py --config "$cfg" --tag train $sm --cont-seed "$seed" \
    --out-name "$name.partial.npz" || return 1
  mv "$root/$name.partial.npz" "$root/$name.npz"
}
NV=$(ys verify_seeds expand.verify.seeds); VSEED=$(y expand.verify.seed)
verify_unit() { label_pass "$CFG" "$S2" "$SMOKE" $(( VSEED + 2 * $1 )) "labels_v$1" "$(gpu "$1")"; }
st_verify() { run_par verify verify_unit "$NV"; }

st_discover() { $PY $C/expand_select.py --config "$CFG" $SMOKE discover; }

NR=$(ys replicate_seeds expand.replicate.seeds); RSEED=$(y expand.replicate.seed)
REP=$BASE/replicate
replicate_unit() { label_pass "$REP/config_replicate.yaml" "$REP/stage2" "" $(( RSEED + 2 * $1 )) "labels_rep_$1" "$(gpu "$1")"; }
st_replicate() { run_par replicate replicate_unit "$NR"; }

st_finalize() { $PY $C/expand_select.py --config "$CFG" $SMOKE finalize; }
st_score_v() { CUDA_VISIBLE_DEVICES=$(gpu 0) $PY $C/expand_score_v.py --config "$CFG" $SMOKE; }
st_realize() { $PY $C/realize_branches.py --src "$BASE/selected" --dst "$BASE/realized"; }

SUF=${SMOKE:+_smoke}
st_pack() {
  local EL; EL=$(y expand.pack.ep_len)
  for pair in "selected:$(y expand.finetune.img_repo)" "realized:$(y expand.finetune.real_frames_repo)"; do
    $PY $P/datagen/pack_closed_loop.py --accepted "$BASE/${pair%%:*}" --repo-id "${pair#*:}$SUF" \
      --ep-len "$EL" --pair-cov "$(y expand.pack.pair_cov)" --include-anchor --overwrite || return 1
  done
}
st_build() { $PY $C/expand_select.py --config "$CFG" $SMOKE build; }

FT_SEED=$(y expand.finetune.seed); FT_STEPS=$(y expand.finetune.steps); FT_DROP=$(y expand.finetune.drop)
FT_ROOT=$BASE/finetune/train
ARMS=(img real ctrl)
arm_repo() {
  case $1 in
    img) echo "$(y expand.finetune.img_repo)_plus_real" ;;
    real) echo "$(y expand.finetune.real_frames_repo)_plus_real" ;;
    ctrl) y expand.finetune.real_repo ;;
  esac
}
arm_ckpt() { local r; r=$(arm_repo "$1"); printf '%s/%s_d%s_s%s/checkpoints/%06d/pretrained_model' "$FT_ROOT" "${r#local/}" "$FT_DROP" "$FT_SEED" "$FT_STEPS"; }
finetune_unit() {
  local arm=${ARMS[$1]} ck run; ck=$(arm_ckpt "$arm"); run=${ck%/checkpoints/*}
  [ -d "$ck" ] && { echo "skip $arm (checkpoint exists)"; return 0; }
  rm -rf "$run"   # lerobot-train refuses an existing output_dir; a partial run restarts from scratch
  CUDA_VISIBLE_DEVICES=$(gpu "$1") OUT_ROOT=$FT_ROOT SAVE_FREQ=$FT_STEPS WANDB_ENABLE=${WANDB_ENABLE:-false} \
    bash $P/scripts/finetune_cl.sh "repo=$(arm_repo "$arm")" "$FT_SEED" "$FT_STEPS" "$FT_DROP" || return 1
  [ -d "$ck" ]
}
st_finetune() { [ -n "$SMOKE" ] && { echo "smoke: no fine-tune"; return 0; }; NPAR=$NG run_par finetune finetune_unit 3; }

EV=$BASE/finetune/eval
eval_unit() {
  local arm=${ARMS[$1]} out=$EV/${ARMS[$1]}
  [ -f "$out/success_steps.npz" ] && { echo "skip $arm (evaluated)"; return 0; }
  rm -rf "$out"; mkdir -p "$out" || return 1
  CUDA_VISIBLE_DEVICES=$(gpu "$1") $PY $P/eval/eval_success_steps.py --policy.path="$(arm_ckpt "$arm")" \
    --policy.device=cuda --policy.use_amp=false --policy.n_action_steps="$(y expand.eval.n_action_steps)" \
    --env.type=pusht --eval.n_episodes="$(y expand.eval.n_episodes)" --eval.batch_size=50 \
    --eval.use_async_envs=false --seed="$(y expand.eval.seed)" --output_dir="$out"
}
st_eval() { [ -n "$SMOKE" ] && { echo "smoke: no eval"; return 0; }; NPAR=$NG run_par eval eval_unit 3; }
st_compare() {
  [ -n "$SMOKE" ] && { echo "smoke: no compare"; return 0; }
  local bd; bd=$(y expand.eval.base_dir); bd=${bd/#\~/$HOME}
  [ -e "$EV/base" ] || ln -s "$bd" "$EV/base" || return 1
  $PY $P/attribution/finetune_compare.py "$EV" --arms base img real ctrl \
    --pairs img:ctrl real:ctrl img:real img:base real:base ctrl:base
}

# ---------------------------------------------------------------- main loop
FINISHED=0
# any exit that is not the normal end (set -u / -e abort, kill) still leaves a FAIL line in status.txt
trap '[ "$FINISHED" = 1 ] || status "FAIL driver exited unexpectedly (code $?) -- see run.log; re-run the same command to resume"' EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM   # route signals (tmux kill, ssh hangup) through EXIT
status "START stages: $STAGES ${SMOKE} gpus=${GPU_LIST[*]}"
for stage in $STAGES; do
  case " $ALL " in *" $stage "*) ;; *) status "FAIL unknown stage $stage"; FINISHED=1; exit 1 ;; esac
  if [ -f "$BASE/.done/$stage" ]; then status "skip $stage (done)"; continue; fi
  status "begin $stage"
  if "st_$stage"; then
    touch "$BASE/.done/$stage"; status "done $stage"
  else
    status "FAIL $stage (see $LOGS and run.log); re-run the same command to resume"; FINISHED=1; exit 1
  fi
done
FINISHED=1
status "ALL DONE"
