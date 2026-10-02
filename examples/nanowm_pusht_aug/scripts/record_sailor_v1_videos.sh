#!/usr/bin/env bash
# SAILOR v1: one full-episode rollout MP4 per (model, seed) for control (base), V-FT and R-FT.
# lerobot-eval seeds episode i with SEED+i, so --seed=S --eval.n_episodes=1 records the episode whose
# start is env.reset(seed=S); all three models therefore face the SAME two starting states.
#
#   bash record_sailor_v1_videos.sh [SEED ...]     (default seeds: 800000 800001)
# Output: examples/nanowm_pusht_aug/output/sailor_v1/video/<model>_seed<S>.mp4 + summary.json
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
SEEDS=("${@:-800000 800001}")
read -r -a SEEDS <<< "${SEEDS[*]}"
NA=10                                   # same as the base policy's eval (finetune/logs/eval_base.log)
DS=$HOME/datastorage/nanowm_pusht_aug
VID=examples/nanowm_pusht_aug/output/sailor_v1/video
WORK=$DS/finetune/eval/videos_work
declare -A CKPT=(
  [control_base]=outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model
  [valuenet_ft]=$DS/finetune/train/pusht_plus_sailorv1_v_d8_s1000/checkpoints/last/pretrained_model
  [random_ft]=$DS/finetune/train/pusht_plus_sailorv1_r_d8_s1000/checkpoints/last/pretrained_model
)
mkdir -p "$VID"
for m in control_base valuenet_ft random_ft; do
  [ -e "${CKPT[$m]}" ] || { echo "missing checkpoint: ${CKPT[$m]}"; exit 1; }
  for s in "${SEEDS[@]}"; do
    out=$WORK/${m}_seed$s
    rm -rf "$out"
    uv run --extra training lerobot-eval \
      --policy.path="${CKPT[$m]}" --policy.device=cuda --policy.use_amp=false \
      --policy.n_action_steps=$NA --env.type=pusht \
      --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false \
      --seed="$s" --output_dir="$out" > "$WORK/${m}_seed$s.log" 2>&1 || { tail -20 "$WORK/${m}_seed$s.log"; exit 1; }
    cp "$out"/videos/pusht_0/eval_episode_0.mp4 "$VID/${m}_seed$s.mp4"
    echo "$m seed=$s -> $VID/${m}_seed$s.mp4"
  done
done
uv run python - "$WORK" "$VID/summary.json" "${SEEDS[@]}" <<'PY'
import json, sys
from pathlib import Path
work, dst, seeds = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
res = {}
for m in ("control_base", "valuenet_ft", "random_ft"):
    for s in seeds:
        mt = json.loads((work / f"{m}_seed{s}" / "eval_info.json").read_text())["per_task"][0]["metrics"]
        res[f"{m}_seed{s}"] = {k: v[0] for k, v in mt.items() if isinstance(v, list) and v and k != "video_paths"}
dst.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
PY
