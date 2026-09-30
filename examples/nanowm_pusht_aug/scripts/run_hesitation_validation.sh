#!/usr/bin/env bash
# Long validation of the hesitation-band hypothesis (cov 0.93-0.95).
#
# Measured motivation: the default trim rule needs >= ep-len genuine pre-crossing actions, i.e.
# cross-j >= 2, which silently drops j = cross-1 -- the frame right before the win. Those excluded
# frames sit at median coverage 0.931 and make up 539 of the 1,013 frames in [0.93,0.95), so 53% of
# the signal in the band where the policy visibly hesitates was being thrown away.
#
# Phase A (separate script, already run): equal-dose 464 comparison of frame selection
#          Trimmed (j<=cross-2) vs Lastpush (j==cross-1) vs Allpre (j<=cross-1).
# Phase B (this script): raise statistical power. Every conclusion so far carries a ~+-7 pp CI at
#          200 episodes, which cannot resolve the 5 pp effects being chased. 1000 episodes gives
#          ~+-3 pp. Re-evaluate the decisive arms at both inference cadences.
set -euo pipefail
# repo root -- derived from this script's own location (examples/nanowm_pusht_aug/scripts/)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=examples/nanowm_pusht_aug
N=1000

declare -A CK=(
  [control]=outputs/train/cl_aug/a_real_only_d8_s1000
  [trimmed]=outputs/train/cl_aug/trimmed464_d8_s1000
  [lastpush]=outputs/train/cl_aug/lastpush464_d8_s1000
  [allpre]=outputs/train/cl_aug/allpre464_d8_s1000
  [verified]=outputs/train/cl_aug/verified464_d8_s1000
)
for arm in control trimmed lastpush allpre verified; do
  for NA in 8 10; do
    out=outputs/eval/cl_aug/${arm}_n${N}_na${NA}
    [ -f "$out/eval_info.json" ] && { echo "skip $out"; continue; }
    uv run --extra training lerobot-eval \
      --policy.path=${CK[$arm]}/checkpoints/last/pretrained_model \
      --policy.n_action_steps=$NA --policy.device=cuda --policy.use_amp=false \
      --env.type=pusht --eval.n_episodes=$N --eval.batch_size=50 --eval.use_async_envs=false \
      --seed=100000 --output_dir=$out
    echo "=== ${arm} na${NA} n${N} DONE ==="
  done
done
echo "=== HIPOWER DONE ==="
