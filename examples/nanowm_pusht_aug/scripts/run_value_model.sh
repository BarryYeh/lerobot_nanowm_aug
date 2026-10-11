#!/usr/bin/env bash
# Value / reward-model (V/RM) pipeline on NanoWM latents.
# Legacy plan: value-model-opus5-5-sonnet5-floofy-engelbart.md (configs/value_model_topband.yaml).
# SAILOR v1: examples/nanowm_pusht_aug/docs/decisions/sailor_v1.md (configs/sailor_v1_reset.yaml).
#
#   bash run_value_model.sh <config.yaml> [stage ...] [--smoke]
#   stages (default: the SAILOR order  collect encode anchors imagine label test_anchors test_imagine
#           test_label train report ; the legacy topband config, which has no stage2.replan_every_rounds,
#           defaults to the old  seeds collect encode imagine label train report  and the old imagine/label):
#     seeds         legacy top-band stage-2 anchors (CoG + body)              eval/make_topband_seeds.py
#     collect       stage-1 base-policy rollouts -> value states              critic/collect_value_states.py
#     encode        stage-1 shards -> shard-aligned SD-VAE latents            imagined_rollout.py encode
#     anchors       stage-2 anchors from reset rollouts (train set)           critic/make_reset_anchors.py
#     imagine       stage-2 NanoWM branches: sigma 0.0 first (writes lead_in.npz), then every other sigma
#                   replaying that lead-in with the same --torch-seed         eval/closed_loop_rollout.py
#     label         stage-2 exact-replay labels (--tag train)                 critic/label_imagined_branches.py
#     test_anchors  G6 test anchors (seed 750000)                             critic/make_reset_anchors.py
#     test_imagine  / test_label  same as imagine / label for the test set (test_stage2/, seeds_test.npz)
#     train         value/oracle ensemble training + health checks            critic/train_value_model.py
#     report        held-out gate report                                      critic/value_gate_report.py
# --smoke keeps EVERY artifact (anchors and test anchors included) under $OUT/smoke/.
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
P=examples/nanowm_pusht_aug
C=$P/critic
E=$P/eval
A=examples/nanowm_pusht_aug/attribution
PY="uv run --extra training python"

y() { uv run python "$A/config_io.py" "$CFG" "$1"; }
# legacy (topband) config: no stage2.replan_every_rounds -> the old stage list / imagine / label behaviour
LEGACY=0; y stage2.replan_every_rounds >/dev/null 2>&1 || LEGACY=1
if [ "$LEGACY" = 1 ]; then DEFAULT_STAGES="seeds collect encode imagine label train report"
else DEFAULT_STAGES="collect encode anchors imagine label test_anchors test_imagine test_label train report"; fi
STAGES="${ARGS[*]:-$DEFAULT_STAGES}"
OUT=$(y out_dir)
# --smoke keeps every artifact under $OUT/smoke/ so it can never clobber or feed the full run
if [ -n "$SMOKE" ]; then BASE=$OUT/smoke; else BASE=$OUT; fi
S1=$BASE/stage1
S2=$BASE/stage2
T2=$BASE/test_stage2
SEEDS_TRAIN=$BASE/seeds_stage2.npz
SEEDS_TEST=$BASE/seeds_test.npz

# imagine_set <stage2 root> <seeds npz> <n seeds> <torch seed> <noise seed>
# sigma runs share the anchors' lead-in: the FIRST sigma (must be 0.0) plays the policy lead-in and writes
# <root>/sigma0.0/work/lead_in.npz, every later sigma replays it (O1: paired candidates, same decision point).
imagine_set() {
  local ROOT=$1 SEEDS=$2 NSEEDS=$3 TSEED=$4 NSEED=$5
  local SIGMAS CKPT SIG RUN LEADARG FIRST=""
  if [ -n "$SMOKE" ]; then K=$(y smoke.stage2_k); else K=$(y stage2.k); fi
  SIGMAS=$(y stage2.inject_sigma | tr -d '[],')
  CKPT=$(y "ckpt.$(y ckpt_key)")
  for SIG in $SIGMAS; do
    if [ -z "$FIRST" ]; then
      [ "$SIG" = "0.0" ] || { echo "stage2.inject_sigma must start with 0.0, got $SIG"; exit 1; }
      FIRST=$SIG; LEADARG=()
    else
      LEADARG=(--lead-in "$ROOT/sigma$FIRST/work/lead_in.npz")
    fi
    RUN=$ROOT/sigma$SIG
    # a symlinked sigma dir belongs to another run (e.g. sailor_v1_s025 reuses sailor_v1/sigma0.0): never wipe it
    if [ -L "$RUN" ]; then echo "refusing to re-imagine $RUN: it is a symlink into another run"; exit 1; fi
    rm -rf "$RUN/branches"; mkdir -p "$RUN/branches"   # work/ is wiped by the driver itself
    $PY $E/closed_loop_rollout.py --ckpt "$CKPT" --seeds "$SEEDS" \
      --n-seeds "$NSEEDS" --k "$K" --h-max "$(y stage2.h_max)" --inject-sigma "$SIG" \
      --accept-cov 0 --accept-mode terminal --save-latents \
      --replan-every-rounds "$(y stage2.replan_every_rounds)" --noise-hold "$(y stage2.noise_hold)" \
      --max-z "$(y stage2.max_z)" --lead-na "$(y stage2.lead_na)" \
      --torch-seed "$TSEED" --noise-seed "$NSEED" ${LEADARG[@]+"${LEADARG[@]}"} \
      --num-sampling-steps "$(y nanowm.num_sampling_steps)" --wm-batch "$(y nanowm.wm_batch)" \
      --work "$RUN/work" --out "$RUN/branches"
  done
}

for stage in $STAGES; do
  echo "=== $stage $(date +%H:%M) ==="
  case "$stage" in
    collect) $PY $C/collect_value_states.py --config "$CFG" $SMOKE ;;
    seeds)  # legacy (topband config only)
      $PY $E/make_topband_seeds.py --config "$CFG" --section stage2.anchors --out-name seeds_stage2.npz ;;
    encode)
      NW=$(y nanowm_python)
      for shard in "$S1"/shard_*.npz; do
        lat="$S1/latents_$(basename "$shard" .npz | sed 's/^shard_//').npz"
        # refuse stale latents: only reuse a latents file that is NEWER than its shard (collect also deletes it)
        if [ -f "$lat" ] && [ "$lat" -nt "$shard" ]; then echo "skip $(basename "$lat") (done, newer than shard)"; continue; fi
        [ -f "$lat" ] && { echo "stale $(basename "$lat") (older than its shard) -> re-encoding"; rm -f "$lat"; }
        "$NW" $P/imagined_rollout.py encode --in "$shard" --out "$lat" --key frames_224 \
          --seed "$(y encode.seed)" --batch "$(y encode.batch)"
      done
      ;;
    anchors)
      $PY $C/make_reset_anchors.py --config "$CFG" --section stage2.anchors --out "$SEEDS_TRAIN" $SMOKE ;;
    test_anchors)
      $PY $C/make_reset_anchors.py --config "$CFG" --section test_anchors --out "$SEEDS_TEST" $SMOKE ;;
    imagine)
      if [ "$LEGACY" = 1 ]; then
        if [ -n "$SMOKE" ]; then NSEEDS=$(y smoke.stage2_anchors); K=$(y smoke.stage2_k); SIGMAS="0.0"
        else NSEEDS=$(y stage2.anchors.n); K=$(y stage2.k); SIGMAS=$(y stage2.inject_sigma | tr -d '[],'); fi
        CKPT=$(y "ckpt.$(y ckpt_key)")
        for SIG in $SIGMAS; do
          RUN=$S2/sigma$SIG
          rm -rf "$RUN/branches"; mkdir -p "$RUN/branches"
          $PY $E/closed_loop_rollout.py --ckpt "$CKPT" --seeds "$OUT/seeds_stage2.npz" \
            --n-seeds "$NSEEDS" --k "$K" --h-max "$(y stage2.h_max)" --inject-sigma "$SIG" \
            --accept-cov 0 --accept-mode terminal --save-latents \
            --num-sampling-steps "$(y nanowm.num_sampling_steps)" --wm-batch "$(y nanowm.wm_batch)" \
            --work "$RUN/work" --out "$RUN/branches"
        done
        continue
      fi
      if [ -n "$SMOKE" ]; then NSEEDS=$(y smoke.stage2_anchors); else NSEEDS=$(y stage2.anchors.n); fi
      imagine_set "$S2" "$SEEDS_TRAIN" "$NSEEDS" "$(y stage2.torch_seed)" "$(y stage2.noise_seed)"
      ;;
    test_imagine)
      if [ -n "$SMOKE" ]; then NSEEDS=$(y smoke.test_anchors); else NSEEDS=$(y test_anchors.n); fi
      imagine_set "$T2" "$SEEDS_TEST" "$NSEEDS" "$(y test_anchors.torch_seed)" "$(y test_anchors.noise_seed)"
      ;;
    label)
      if [ "$LEGACY" = 1 ]; then SEEDS_TRAIN=$OUT/seeds_stage2.npz; fi
      $PY $C/label_imagined_branches.py --config "$CFG" --tag train --seeds "$SEEDS_TRAIN" $SMOKE ;;
    test_label) $PY $C/label_imagined_branches.py --config "$CFG" --tag test --seeds "$SEEDS_TEST" $SMOKE ;;
    train) $PY $C/train_value_model.py --config "$CFG" $SMOKE ;;
    report) $PY $C/value_gate_report.py --config "$CFG" $SMOKE ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
echo "=== run_value_model.sh done $(date +%H:%M) ==="
