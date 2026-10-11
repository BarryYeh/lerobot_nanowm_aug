#!/usr/bin/env bash
# SAILOR v1-s025 (configs/sailor_v1_s025.yaml): sailor_v1 with inject_sigma 0.1 replaced by 0.25.
#
#   bash run_sailor_v1_s025.sh [stage ...]      stages (default: all, in order):
#     setup     out_dir with stage1/, seeds_*.npz and stage2|test_stage2/sigma0.0 symlinked to sailor_v1
#     imagine   train branches for sigma 0.25 only, replaying sailor_v1's sigma0.0 lead-in
#     test_imagine   same for the G6 test anchors
#     rest      run_value_model.sh label test_label train report (labels BOTH sigmas in one batch, O9)
#
# The sigma0.25 run uses sailor_v1's stage2.torch_seed / noise_seed, so every candidate shares its anchor,
# lead-in and round-0 base chunk with sailor_v1's sigma0.0 candidate, and its noise draws are v1's sigma0.1
# draws scaled by 2.5 (common random numbers). label asserts the shared round-0 chunk across sigma runs.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/sailor_v1_s025.yaml
P=examples/nanowm_pusht_aug
PY="uv run --extra training python"
y() { uv run python "$P/attribution/config_io.py" "$CFG" "$1"; }

OUT=$(y out_dir)
V1=$(dirname "$OUT")/sailor_v1
SIG=$(y stage2.inject_sigma | tr -d '[],' | awk '{print $2}')   # the non-zero sigma
[ "$SIG" = "0.25" ] || { echo "expected inject_sigma [0.0, 0.25], got $(y stage2.inject_sigma)"; exit 1; }
STAGES="${*:-setup imagine test_imagine rest}"

# imagine_one <stage2 root> <seeds npz> <n seeds> <torch seed> <noise seed>
imagine_one() {
  local ROOT=$1 SEEDS=$2 NSEEDS=$3 TSEED=$4 NSEED=$5 RUN
  RUN=$ROOT/sigma$SIG
  [ -L "$RUN" ] && { echo "refusing: $RUN is a symlink"; exit 1; }
  [ -L "$ROOT/sigma0.0" ] || { echo "$ROOT/sigma0.0 must be the symlink to sailor_v1 (run setup)"; exit 1; }
  rm -rf "$RUN/branches"; mkdir -p "$RUN/branches"
  $PY $P/eval/closed_loop_rollout.py --ckpt "$(y "ckpt.$(y ckpt_key)")" --seeds "$SEEDS" \
    --n-seeds "$NSEEDS" --k "$(y stage2.k)" --h-max "$(y stage2.h_max)" --inject-sigma "$SIG" \
    --accept-cov 0 --accept-mode terminal --save-latents \
    --replan-every-rounds "$(y stage2.replan_every_rounds)" --noise-hold "$(y stage2.noise_hold)" \
    --max-z "$(y stage2.max_z)" --lead-na "$(y stage2.lead_na)" \
    --torch-seed "$TSEED" --noise-seed "$NSEED" --lead-in "$ROOT/sigma0.0/work/lead_in.npz" \
    --num-sampling-steps "$(y nanowm.num_sampling_steps)" --wm-batch "$(y nanowm.wm_batch)" \
    --work "$RUN/work" --out "$RUN/branches"
}

for stage in $STAGES; do
  echo "=== $stage $(date +%H:%M) ==="
  case "$stage" in
    setup)
      mkdir -p "$OUT/stage2" "$OUT/test_stage2"
      for f in stage1 seeds_stage2.npz seeds_test.npz stage2/sigma0.0 test_stage2/sigma0.0; do
        [ -e "$V1/$f" ] || { echo "missing $V1/$f"; exit 1; }
        if [ -L "$OUT/$f" ]; then continue; fi
        [ -e "$OUT/$f" ] && { echo "$OUT/$f exists and is not a symlink"; exit 1; }
        ln -s "$V1/$f" "$OUT/$f"
      done
      ls -l "$OUT" "$OUT/stage2" "$OUT/test_stage2" ;;
    imagine)
      imagine_one "$OUT/stage2" "$OUT/seeds_stage2.npz" "$(y stage2.anchors.n)" \
        "$(y stage2.torch_seed)" "$(y stage2.noise_seed)" ;;
    test_imagine)
      imagine_one "$OUT/test_stage2" "$OUT/seeds_test.npz" "$(y test_anchors.n)" \
        "$(y test_anchors.torch_seed)" "$(y test_anchors.noise_seed)" ;;
    rest)
      bash $P/scripts/run_value_model.sh "$CFG" label test_label train report ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
echo "=== run_sailor_v1_s025.sh done $(date +%H:%M) ==="
