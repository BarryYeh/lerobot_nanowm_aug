#!/usr/bin/env bash
# CUPID filtering of NanoWM imagined pairs (setting B), end to end. The config is the only input.
#
#   bash research/nanowm_aug/proj/pusht/cupid/run_cupid_filter.sh <config.yaml> [stage ...]
#   stages (default: all, in this order):
#     record     Stage 2  -- test rollouts of theta              -> <out>/test_rollouts.npz
#     featurize  Stage 3  -- projected per-sample gradients      -> <out>/features/
#     replicate  fail-fast -- featurize again with seeds + 1     -> <out>/features_r1/
#     score      Stage 3  -- pair scores + diagnostics           -> <out>/pair_scores.csv, score_summary.json
#     build      Stage 4  -- cupid/random filtered datasets      -> local/<train_repo_id>_{cupid,random}_r{f}
#     finetune   Stage 4  -- re-fine-tune from the base policy   -> outputs/train/cl_aug/<tag>_d<drop>_s<seed>
#     eval       n-episode eval of every arm + the references    -> <out>/eval/<tag>/
#     report     success table + paired CUPID-vs-Random / vs-Control -> <out>/report.md
#
# Fail-fast gates: stop after `score` if spearman_vs_features_r1 < 0.8 (scores are seed noise) or
# if the XTX condition number is huge with lambda_reg = 0.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RES=research/nanowm_aug  # the one PYTHONPATH root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="${1:?usage: run_cupid_filter.sh <config.yaml> [stage ...]}"
shift
STAGES="${*:-record featurize replicate score build finetune eval report}"
PY="uv run --extra training python"

y() {  # y a.b.c -> config value, with `extends:` resolved (config_io has no lerobot imports)
  uv run python "$RES/nwaug/attribution/config_io.py" "$CFG" "$1"
}
OUT=$(y out_dir)
DROP=$(y drop_n_last_frames)
SEED=$(y finetune.seed)
STEPS=$(y finetune.steps)
REFS="a_real_only untrimmed464"  # existing arms, re-evaluated under the same protocol

arms() { $PY $RES/nwaug/attribution/build_filtered_datasets.py --config "$CFG" --list; }
ckpt_of() { echo "outputs/train/cl_aug/$1_d${DROP}_s${SEED}/checkpoints/last/pretrained_model"; }

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
  echo "=== $stage ==="
  case "$stage" in
    record)    $PY $RES/proj/pusht/cupid/record_policy_rollouts.py --config "$CFG" ;;
    featurize) $PY $RES/nwaug/attribution/featurize.py --config "$CFG" ;;
    replicate) $PY $RES/nwaug/attribution/featurize.py --config "$CFG" --replicate 1 ;;
    score)     uv run --extra training --with matplotlib python $RES/nwaug/attribution/score_pairs.py --config "$CFG" ;;
    build)     $PY $RES/nwaug/attribution/build_filtered_datasets.py --config "$CFG" ;;
    finetune)
      for repo in $(arms); do
        SAVE_FREQ=$(y finetune.save_freq) bash $RES/proj/pusht/scripts/finetune_cl.sh "repo=$repo" "$SEED" "$STEPS" "$DROP"
      done ;;
    eval)
      for tag in $REFS $(arms | sed 's#^local/##'); do
        [ -f "$OUT/eval/$tag/eval_info.json" ] && { echo "skip $tag (done)"; continue; }
        eval_pusht "$(ckpt_of "$tag")" "$OUT/eval/$tag" "$(y eval.n_episodes)" "$(y eval.seed)" "$(y eval.n_action_steps)"
      done ;;
    report)    uv run python $RES/nwaug/attribution/paired_report.py "$OUT" --ref a_real_only ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
