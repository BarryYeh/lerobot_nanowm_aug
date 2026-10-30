#!/usr/bin/env bash
# Stage 2 (add-one-in / CUPID demo SELECTION) end to end: generate band-start imagined pool ->
# score with theta=base policy (pool as a HOLDOUT, never in XtX) -> select top-k -> fine-tune -> eval.
# Waits for stage 1's GPU job to finish (single GPU box) before starting its own generation.
#
#   bash run_cupid_select.sh <config.yaml> [stage ...]
#   stages (default: all, in order):
#     wait_stage1  block until research/nanowm_aug/logs/stage1_run.log shows "exit="
#     seeds        band starts (pool/test/eval, disjoint)          -> <out>/seeds_{pool,test,eval}.npz
#     generate     closed_loop_rollout.py per injection level       -> <out>/pool_raw/sigma<k>/
#     merge        symlink both levels into one dir for the packer -> <out>/pool_raw/merged/
#     pack         anchor + imagined pairs, no coverage floor       -> local/<pool_repo_id>
#     record       test rollouts from band starts (base policy)     -> <out>/test_rollouts.npz
#     featurize    theta=base; pool scored but held OUT of XtX      -> <out>/features{,_r1}/
#     score        pair scores + diagnostics (source/sigma groups)  -> <out>/pair_scores.csv
#     build        real + top-k (cupid) / real + k random           -> local/pusht_plus_..._k500
#     finetune     both arms, from the base policy
#     eval         band (400, paired) + standard random-reset (1000, safety check)
#     report       success table + paired CUPID-vs-Random
#
# Unlike stage 1, a failed rank-stability gate does NOT stop this script (the deadline fallback in
# the plan): it proceeds with the 2-replicate mean and marks the report "reliability uncertain".
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RES=research/nanowm_aug  # the one PYTHONPATH root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="${1:?usage: run_cupid_select.sh <config.yaml> [stage ...]}"
shift
STAGES="${*:-wait_stage1 seeds generate merge pack record featurize score build finetune eval report}"
PY="uv run --extra training python"

y() { uv run python "$RES/nwaug/attribution/config_io.py" "$CFG" "$1"; }
OUT=$(y out_dir)
DROP=$(y drop_n_last_frames)
SEED=$(y finetune.seed)
STEPS=$(y finetune.steps)
K=$(y select.k)
CKPT_BASE=$(y ckpt)
POOL_REPO=$(y pool_repo_id)
RAW=$OUT/pool_raw

ckpt_of() { echo "outputs/train/cl_aug/$1_d${DROP}_s${SEED}/checkpoints/last/pretrained_model"; }
arms() { $PY $RES/nwaug/attribution/build_selected_datasets.py --config "$CFG" --list; }

eval_pusht() {  # <ckpt> <out> <n> <seed> [n_action_steps]
  local ckpt="$1" out="$2" n="$3" seed="$4" na="${5:-}"
  mkdir -p "$out"
  echo "eval $ckpt  n_episodes=$n  seed=$seed  n_action_steps=${na:-ckpt} -> $out"
  uv run --extra training lerobot-eval \
    --policy.path="$ckpt" \
    --policy.device=cuda \
    --policy.use_amp=false \
    ${na:+--policy.n_action_steps="$na"} \
    --env.type=pusht \
    --eval.n_episodes="$n" \
    --eval.batch_size=50 \
    --eval.use_async_envs=false \
    --seed="$seed" \
    --output_dir="$out" 2>&1 | tee "$out/eval.log"
}

for stage in $STAGES; do
  echo "=== $stage $(date +%H:%M) ==="
  case "$stage" in
    wait_stage1)
      LOG=research/nanowm_aug/logs/stage1_run.log
      while [ ! -f "$LOG" ] || ! grep -aq "^exit=" "$LOG"; do sleep 30; done
      echo "stage1 finished: $(grep -a '^exit=' "$LOG")"
      ;;
    seeds) $PY $RES/nwaug/wm/make_topband_seeds.py --config "$CFG" ;;
    generate)
      NSEEDS=$(y topband.pool.n)
      for SIG in $(y generate.inject_sigma | tr -d '[],'); do
        WORK=$RAW/work_sigma$SIG; OUTB=$RAW/sigma$SIG
        rm -rf "$WORK" "$OUTB"; mkdir -p "$OUTB"
        $PY $RES/nwaug/wm/closed_loop_rollout.py --ckpt "$CKPT_BASE" \
          --seeds "$OUT/seeds_pool.npz" --n-seeds "$NSEEDS" --k "$(y generate.k)" --h-max "$(y generate.h_max)" \
          --inject-sigma "$SIG" --max-z "$(y generate.max_z)" --noise-hold "$(y generate.noise_hold)" \
          --noise-seed "$(y generate.noise_seed)" --accept-cov 0 --accept-mode terminal \
          --num-sampling-steps "$(y nanowm.num_sampling_steps)" --wm-batch "$(y nanowm.wm_batch)" \
          --work "$WORK" --out "$OUTB"
      done
      ;;
    merge)
      MERGED=$RAW/merged; rm -rf "$MERGED"; mkdir -p "$MERGED"
      for d in "$RAW"/sigma*/; do
        sig=$(basename "$d")
        for f in "$d"*.npz; do [ -e "$f" ] && ln -s "$(realpath "$f")" "$MERGED/${sig}_$(basename "$f")"; done
      done
      echo "merged $(ls "$MERGED" | wc -l) branches -> $MERGED"
      ;;
    pack)
      rm -rf "${HF_LEROBOT_HOME:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/lerobot}/$POOL_REPO"
      $PY $RES/nwaug/datagen/pack_closed_loop.py --accepted "$RAW/merged" --repo-id "$POOL_REPO" \
        --pair-cov 0.0 --ep-len "$(y pack.ep_len)" --include-anchor --overwrite
      ;;
    record) $PY $RES/proj/pusht/cupid/record_policy_rollouts.py --config "$CFG" ;;
    featurize) $PY $RES/nwaug/attribution/featurize.py --config "$CFG" ;;
    score)
      uv run --extra training --with matplotlib python $RES/nwaug/attribution/score_pairs.py --config "$CFG"
      read -r RHO R_AVG < <(uv run --no-sync python - "$OUT/score_summary.json" <<'PY'
import json, math, sys
from pathlib import Path
d = json.load(open(sys.argv[1]))
rho = d.get("spearman_features_vs_features_r1", float("nan"))
if isinstance(rho, float) and math.isnan(rho):
    r_avg = float("nan")
else:
    try:
        r_avg = 2 * rho / (1 + rho)
    except (TypeError, ZeroDivisionError):
        r_avg = float("nan")
print(rho, r_avg)
PY
)
      echo "$R_AVG" > "$OUT/gate_r_avg.txt"
      echo "gate: single-replicate rho=$RHO  2-replicate mean reliability=$R_AVG (threshold 0.8; NOT blocking here)"
      ;;
    build) $PY $RES/nwaug/attribution/build_selected_datasets.py --config "$CFG" ;;
    finetune)
      for repo in $(arms); do
        SAVE_FREQ=$(y finetune.save_freq) WANDB_ENTITY=$(y wandb.entity) WANDB_PROJECT=$(y wandb.project) \
          WANDB_MODE=$(y wandb.mode) WANDB_ENABLE=${WANDB_ENABLE:-$(y wandb.enable | tr '[:upper:]' '[:lower:]')} \
          bash $RES/proj/pusht/scripts/finetune_cl.sh "repo=$repo" "$SEED" "$STEPS" "$DROP"
      done
      ;;
    eval)
      NA=$(y band_eval.n_action_steps)
      SELECT_TAGS="$(arms | sed 's#^local/##')"
      for tag in $SELECT_TAGS base a_real_only; do
        [ -f "$OUT/eval/$tag/successes.json" ] && { echo "skip $tag (done)"; continue; }
        C=$([ "$tag" = base ] && echo "$CKPT_BASE" || ckpt_of "$tag")
        $PY $RES/proj/pusht/cupid/band_eval.py --ckpt "$C" --states "$OUT/$(y band_eval.start_states)" \
          --n "$(y band_eval.n_episodes)" --n-action-steps "$NA" --seed 0 --out "$OUT/eval/$tag"
      done
      # safety check: standard random-reset eval, same protocol as stage 1 (n/seed/na all shared
      # via extends: common.yaml, so it's a fair comparison against stage 1's numbers)
      mkdir -p "$OUT/eval_random"
      STAGE1_REF=outputs/cupid/untrimmed464/eval/a_real_only
      if [ -f "$STAGE1_REF/eval_info.json" ] && [ ! -e "$OUT/eval_random/a_real_only" ]; then
        ln -s "$(realpath "$STAGE1_REF")" "$OUT/eval_random/a_real_only"
        echo "reused stage 1's a_real_only random-reset eval"
      fi
      for tag in $SELECT_TAGS a_real_only; do
        [ -f "$OUT/eval_random/$tag/eval_info.json" ] && { echo "skip $tag random (done)"; continue; }
        eval_pusht "$(ckpt_of "$tag")" "$OUT/eval_random/$tag" "$(y eval.n_episodes)" "$(y eval.seed)" "$(y eval.n_action_steps)"
      done
      ;;
    report)
      uv run python $RES/nwaug/attribution/paired_report.py "$OUT" --ref base --report report_band.md
      uv run python $RES/nwaug/attribution/paired_report.py "$OUT" --eval-subdir eval_random --ref a_real_only --report report_random.md
      R_AVG=$(cat "$OUT/gate_r_avg.txt" 2>/dev/null || echo "?")
      echo "stage2 gate R_avg=$R_AVG (>= 0.8 wanted)" | tee -a "$OUT/report_band.md"
      ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
echo "=== run_cupid_select.sh done $(date +%H:%M) ==="
