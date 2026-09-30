#!/usr/bin/env bash
# Value / reward-model (V/RM) pipeline on NanoWM latents, top-wall band.
# Plan: value-model-opus5-5-sonnet5-floofy-engelbart.md.
#
#   bash run_value_model.sh <config.yaml> [stage ...] [--smoke]
#   stages (default: all, in order):
#     seeds     stage-2 anchor states (CoG + body)                     eval/make_topband_seeds.py
#     collect   stage-1 base-policy rollouts -> value states           critic/collect_value_states.py
#     encode    stage-1 shards -> shard-aligned SD-VAE latents         imagined_rollout.py encode
#     imagine   stage-2 NanoWM branches, one run per inject_sigma      eval/closed_loop_rollout.py
#     label     stage-2 exact-replay labels                            critic/label_imagined_branches.py
#     train     value/oracle ensemble training + health checks         critic/train_value_model.py
#     report    held-out gate report (G1-G5 + diagnostics)             critic/value_gate_report.py
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="${1:?usage: run_value_model.sh <config.yaml> [stage ...] [--smoke]}"
shift
SMOKE=""
ARGS=()
for a in "$@"; do
  if [ "$a" = "--smoke" ]; then SMOKE="--smoke"; else ARGS+=("$a"); fi
done
STAGES="${ARGS[*]:-seeds collect encode imagine label train report}"
P=examples/nanowm_pusht_aug
C=$P/critic
E=$P/eval
A=examples/nanowm_pusht_aug/attribution
PY="uv run --extra training python"

y() { uv run python "$A/config_io.py" "$CFG" "$1"; }
OUT=$(y out_dir)
# --smoke keeps every artifact under $OUT/smoke/ so it can never clobber or feed the full run
if [ -n "$SMOKE" ]; then S1=$OUT/smoke/stage1; S2=$OUT/smoke/stage2; else S1=$OUT/stage1; S2=$OUT/stage2; fi

for stage in $STAGES; do
  echo "=== $stage $(date +%H:%M) ==="
  case "$stage" in
    collect) $PY $C/collect_value_states.py --config "$CFG" $SMOKE ;;
    seeds)
      $PY $E/make_topband_seeds.py --config "$CFG" --section stage2.anchors --out-name seeds_stage2.npz ;;
    encode)
      NW=$(y nanowm_python)
      for shard in "$S1"/shard_*.npz; do
        lat="$S1/latents_$(basename "$shard" .npz | sed 's/^shard_//').npz"
        [ -f "$lat" ] && { echo "skip $(basename "$lat") (done)"; continue; }
        "$NW" $P/imagined_rollout.py encode --in "$shard" --out "$lat" --key frames_224 \
          --seed "$(y encode.seed)" --batch "$(y encode.batch)"
      done
      ;;
    imagine)
      if [ -n "$SMOKE" ]; then
        NSEEDS=$(y smoke.stage2_anchors); K=$(y smoke.stage2_k); SIGMAS="0.0"
      else
        NSEEDS=$(y stage2.anchors.n); K=$(y stage2.k); SIGMAS=$(y stage2.inject_sigma | tr -d '[],')
      fi
      CKPT=$(y "ckpt.$(y ckpt_key)")
      for SIG in $SIGMAS; do
        RUN=$S2/sigma$SIG
        rm -rf "$RUN/branches"; mkdir -p "$RUN/branches"   # work/ is wiped by the driver itself
        $PY $E/closed_loop_rollout.py --ckpt "$CKPT" --seeds "$OUT/seeds_stage2.npz" \
          --n-seeds "$NSEEDS" --k "$K" --h-max "$(y stage2.h_max)" --inject-sigma "$SIG" \
          --accept-cov 0 --accept-mode terminal --save-latents \
          --num-sampling-steps "$(y nanowm.num_sampling_steps)" --wm-batch "$(y nanowm.wm_batch)" \
          --work "$RUN/work" --out "$RUN/branches"
      done
      ;;
    label) $PY $C/label_imagined_branches.py --config "$CFG" $SMOKE ;;
    train) $PY $C/train_value_model.py --config "$CFG" $SMOKE ;;
    report) $PY $C/value_gate_report.py --config "$CFG" $SMOKE ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
echo "=== run_value_model.sh done $(date +%H:%M) ==="
