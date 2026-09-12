#!/usr/bin/env bash
# Final action-labelling test: ep_len=10 / drop=9, hold-padding, execution 5 and 16.
#
# Goal: does the NanoWM closed-loop rollout give a performance improvement when the label and
# the deployment cadence are aligned? n_action_steps=5 makes one execution block equal exactly
# one NanoWM round, so the data cadence and the deployment cadence match; na=16 executes the
# whole chunk from a single inference. Both are tested even though na=5 scored poorly on the
# baseline sweep -- what matters is whether NanoWM data helps under a cadence it can match.
#
# Padding: 'hold' writes the agent's own position, the only value that actually commands the PD
# controller to stop. Zero would be an absolute target at the top-left corner; copy keeps pushing.
# With ep_len=10 the auto-pad of L10-L15 copies L9, which is already the hold value, so all 16
# slots are consistent.
#
# drop=9 (not the baseline's 8) is required by ep_len=10, so CONTROL IS RETRAINED AT drop=9 too --
# otherwise the arms would differ in how the real data is sampled, not just in the imagined data.
set -euo pipefail
cd /home/barryyeh/repos/lerobot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=examples/nanowm_pusht_aug

uv run --extra training python -c "
from pathlib import Path; from lerobot.datasets.aggregate import aggregate_datasets
import shutil, json
H=Path.home()/'.cache/huggingface/lerobot'
real=json.load(open(H/'local/pusht_imgonly/meta/stats.json'))
for aug,out in (('local/pusht_cl_untrimmed_h10','local/pusht_plus_untrimmed_h10'),
                ('local/pusht_cl_allpre_h10','local/pusht_plus_allpre_h10')):
    o=H/out
    if o.exists(): shutil.rmtree(o)
    aggregate_datasets(repo_ids=['local/pusht_imgonly',aug], aggr_repo_id=out,
        roots=[H/'local/pusht_imgonly', H/aug], aggr_root=o)
    sp=o/'meta/stats.json'; d=json.load(open(sp))
    for k in ('observation.image','action'): d[k]=real[k]
    json.dump(d, open(sp,'w')); print(out,'ready')
"
echo '=== AGGREGATE DONE ==='

for arm in a untrimmed_h allpre_h; do
  case $arm in
    a)           tag=a_real_only ;;
    untrimmed_h) tag=untrimmed_h10 ;;
    allpre_h)    tag=allpre_h10 ;;
  esac
  rm -rf outputs/train/cl_aug/${tag}_d9_s1000
  bash $P/eval/finetune_cl.sh $arm 1000 20000 9
  for NA in 5 16; do
    uv run --extra training lerobot-eval \
      --policy.path=outputs/train/cl_aug/${tag}_d9_s1000/checkpoints/last/pretrained_model \
      --policy.n_action_steps=$NA --policy.device=cuda --policy.use_amp=false \
      --env.type=pusht --eval.n_episodes=1000 --eval.batch_size=50 --eval.use_async_envs=false \
      --seed=100000 --output_dir=outputs/eval/cl_aug/${tag}_d9_n1000_na$NA
  done
  echo "=== ${tag} DONE ==="
done
echo '=== PAD HOLD ALL DONE ==='
