#!/usr/bin/env bash
# Headline eval: baseline_a_obs2 (a) vs aug_obs2_r1500 (a+aug), last checkpoint,
# on both initial-state distributions. ~2.5-3 h on one 4090 (shares GPU if the
# aug training is still running).
#
#   bash run_eval_suite.sh [ROLLOUTS_PER_INIT] [N_RANDOM]
#   defaults: 5  100
#
# Fail-fast before the full run:
#   uv run --extra training python examples/nanowm_pusht_aug/eval/eval_baseline.py \
#     outputs/train/nanowm_aug/baseline_a_obs2_s100000/checkpoints/last/pretrained_model \
#     --val-inits $VAL --rollouts-per-init 1 --out outputs/eval/nanowm_aug/_ff.json
#   -> mean max_cov should be ~0.2-0.4 (near the random-reset number); wildly off => reset-path bug.
set -euo pipefail

RPI="${1:-5}"
NRAND="${2:-100}"
VAL="$HOME/Downloads/bmw48-osfstorage-datasets-archive/pusht_noise/val/states.pth"
OUT=outputs/eval/nanowm_aug
CKPT_A=outputs/train/nanowm_aug/baseline_a_obs2_s100000/checkpoints/last/pretrained_model
CKPT_B=outputs/train/nanowm_aug/aug_obs2_r1500_s100000/checkpoints/last/pretrained_model

mkdir -p "$OUT"
run() {  # name ckpt
  uv run --extra training python examples/nanowm_pusht_aug/eval/eval_baseline.py "$2" \
      --val-inits "$VAL" --rollouts-per-init "$RPI" --seed 0 --out "$OUT/$1_valinit.json" \
      --save-videos "$OUT/videos" --videos-per-config 1
  uv run --extra training python examples/nanowm_pusht_aug/eval/eval_baseline.py "$2" \
      --random-inits "$NRAND" --seed 0 --out "$OUT/$1_random.json" \
      --save-videos "$OUT/videos" --videos-per-config 1
}

run a   "$CKPT_A"
run aug "$CKPT_B"

uv run --with matplotlib python examples/nanowm_pusht_aug/eval/compare_eval.py \
    "$OUT/a_valinit.json" "$OUT/aug_valinit.json" --out "$OUT/compare_valinit"
uv run --with matplotlib python examples/nanowm_pusht_aug/eval/compare_eval.py \
    "$OUT/a_random.json" "$OUT/aug_random.json" --out "$OUT/compare_random"

echo "== done: $OUT/compare_{valinit,random}_report.txt + _cdf.png =="
