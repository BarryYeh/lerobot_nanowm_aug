#!/usr/bin/env bash
# CUPID filtering of NanoWM imagined pairs (setting B), end to end. The config is the only input.
#
#   bash examples/nanowm_pusht_aug/scripts/run_cupid_filter.sh <config.yaml> [stage ...]
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
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="${1:?usage: run_cupid_filter.sh <config.yaml> [stage ...]}"
shift
STAGES="${*:-record featurize replicate score build finetune eval report}"
A=examples/nanowm_pusht_aug/attribution
E=examples/nanowm_pusht_aug/eval
S=examples/nanowm_pusht_aug/scripts
PY="uv run --extra training python"

y() {  # y a.b.c -> config value, with `extends:` resolved (config_io has no lerobot imports)
  uv run python "$A/config_io.py" "$CFG" "$1"
}
OUT=$(y out_dir)
DROP=$(y drop_n_last_frames)
SEED=$(y finetune.seed)
STEPS=$(y finetune.steps)
REFS="a_real_only untrimmed464"  # existing arms, re-evaluated under the same protocol

arms() { $PY $A/build_filtered_datasets.py --config "$CFG" --list; }
ckpt_of() { echo "outputs/train/cl_aug/$1_d${DROP}_s${SEED}/checkpoints/last/pretrained_model"; }

for stage in $STAGES; do
  echo "=== $stage ==="
  case "$stage" in
    record)    $PY $E/record_policy_rollouts.py --config "$CFG" ;;
    featurize) $PY $A/featurize.py --config "$CFG" ;;
    replicate) $PY $A/featurize.py --config "$CFG" --replicate 1 ;;
    score)     uv run --extra training --with matplotlib python $A/score_pairs.py --config "$CFG" ;;
    build)     $PY $A/build_filtered_datasets.py --config "$CFG" ;;
    finetune)
      for repo in $(arms); do
        SAVE_FREQ=$(y finetune.save_freq) bash $S/finetune_cl.sh "repo=$repo" "$SEED" "$STEPS" "$DROP"
      done ;;
    eval)
      for tag in $REFS $(arms | sed 's#^local/##'); do
        [ -f "$OUT/eval/$tag/eval_info.json" ] && { echo "skip $tag (done)"; continue; }
        bash $S/proto_eval_pusht.sh "$(ckpt_of "$tag")" "$OUT/eval/$tag" "$(y eval.n_episodes)" "$(y eval.seed)" "$(y eval.n_action_steps)"
      done ;;
    report)    uv run python $A/paired_report.py "$OUT" --ref a_real_only ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
