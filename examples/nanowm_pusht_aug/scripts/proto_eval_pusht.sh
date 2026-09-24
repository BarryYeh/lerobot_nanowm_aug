#!/usr/bin/env bash
# Proto: closed-loop eval of an image-only PushT diffusion policy via stock lerobot-eval.
#
# The fps-2 custom control loop (eval_baseline.py) is not needed here: the policy is
# trained at fps 10 and gym_pusht runs at fps 10, so lerobot-eval drives it correctly.
# gym_pusht success_threshold defaults to 0.95 (coverage). lerobot-eval sets
# start_seed=SEED and episode i -> seed SEED+i (deterministic), so two runs with the
# same SEED and N_EPISODES face identical initial states -> pair by episode order.
#
# use_async_envs=false: the async vector env crashes on startup on this box
# (ConnectionResetError in _check_spaces). Sync envs are slower but reliable.
#
#   bash proto_eval_pusht.sh <ckpt_dir> <out_dir> [N_EPISODES] [SEED] [N_ACTION_STEPS]
#   defaults: N_EPISODES=200  SEED=100000  N_ACTION_STEPS=<the checkpoint's own>
set -euo pipefail

CKPT="$1"
OUT="$2"
N="${3:-200}"
SEED="${4:-100000}"
NA="${5:-}"
mkdir -p "$OUT"

echo "eval $CKPT  n_episodes=$N  seed=$SEED  n_action_steps=${NA:-ckpt} -> $OUT"

uv run --extra training lerobot-eval \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --policy.use_amp=false \
  ${NA:+--policy.n_action_steps="$NA"} \
  --env.type=pusht \
  --eval.n_episodes="$N" \
  --eval.batch_size=50 \
  --eval.use_async_envs=false \
  --seed="$SEED" \
  --output_dir="$OUT" 2>&1 | tee "$OUT/eval.log"

echo "wrote $OUT/eval_info.json"
