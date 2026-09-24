#!/usr/bin/env bash
# Equal-dose comparison: Untrimmed / Trimmed / Verified all at EXACTLY 500 pairs.
#
# Why: Verified (161) beat Trimmed (503) by +14.0 pp, but they differed in BOTH label precision
# (100% vs 47.5%) AND dose (161 vs 503). Holding dose constant at 500 isolates label quality.
# 500 is the operating point where an effect is already known to be detectable (Trimmed showed
# a significant -19.5 pp at 503), unlike 161/322 where differences may vanish into noise.
#
# Verified is the bottleneck: it costs 14.2 branches per pair (vs 2.4 for Untrimmed) because a
# branch must BOTH cross 0.95 in the real simulator (47.5% pass) and leave >=9 pre-crossing
# actions. 500 Verified pairs therefore needs ~7,082 branches total.
set -euo pipefail
# repo root -- derived from this script's own location (examples/nanowm_pusht_aug/eval/)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model
P=examples/nanowm_pusht_aug

# ---------- 1. top up the branch corpus ----------
uv run --extra training python $P/eval/proto_make_seeds.py --ckpt $CKPT \
  --episodes 320 --n-seeds 820 --cov-enter 0.95 --lookback 20 --seed 2 \
  --out $P/cl/seeds_endgame3.npz
uv run --no-sync python -c "
import numpy as np
d=np.load('$P/cl/seeds_endgame3.npz'); k=d['meta'][:,2]>=0.80
st,fr,me=d['states'][k],d['frames_224'][k],d['meta'][k]
n=len(st); per=(n+3)//4
for i in range(4):
    sl=slice(i*per,(i+1)*per)
    np.savez_compressed(f'$P/cl/seeds3_part{i}.npz', states=st[sl], frames_224=fr[sl], meta=me[sl])
    print(f'part{i}: {len(st[sl])} seeds')
"
echo '=== SEEDS3 DONE ==='
for i in 0 1 2 3; do
  rm -rf $P/cl/accepted_s3_$i
  uv run --extra training python $P/eval/closed_loop_rollout.py --ckpt $CKPT \
    --seeds $P/cl/seeds3_part$i.npz --n-seeds 250 --k 4 --h-max 8 \
    --num-sampling-steps 30 --wm-batch 24 --accept-mode pair --accept-cov 0.95 \
    --work $P/cl/work_s3_$i --out $P/cl/accepted_s3_$i
  echo "=== ROLLOUT s3_$i DONE ==="
done

# ---------- 2. pool every accepted branch ----------
rm -rf $P/cl/pool500 && mkdir -p $P/cl/pool500
for d in $P/cl/accepted_scale_1 $P/cl/accepted_scale_2 $P/cl/accepted_scale_3 \
         $P/cl/accepted_scale_4 $P/cl/accepted_scale_5 $P/cl/accepted_scale_6 \
         $P/cl/accepted_s3_0 $P/cl/accepted_s3_1 $P/cl/accepted_s3_2 $P/cl/accepted_s3_3; do
  [ -d "$d" ] || continue
  tag=$(basename $d)
  for f in $d/*.npz; do [ -e "$f" ] || continue; ln -s "$(realpath $f)" "$P/cl/pool500/${tag}_$(basename $f)"; done
done
echo "pooled branches: $(ls $P/cl/pool500/*.npz | wc -l)"

# ---------- 3. oracle labels: replay every pooled branch in the real simulator ----------
uv run --extra training python $P/eval/proto_filter_roc.py --branches $P/cl/pool500 \
  --out $P/diag/filter_roc_pool500.npz
echo '=== ORACLE LABELS DONE ==='

# ---------- 4. pack all three arms at EXACTLY 500 pairs ----------
uv run --extra training python $P/datagen/pack_closed_loop.py --accepted $P/cl/pool500 \
  --repo-id local/pusht_cl_untrimmed500 --pair-cov 0.95 --ep-len 9 --max-pairs 500 --overwrite
uv run --extra training python $P/datagen/pack_closed_loop.py --accepted $P/cl/pool500 \
  --repo-id local/pusht_cl_trimmed500 --truncate-cov 0.95 --ep-len 9 --max-pairs 500 --overwrite
uv run --extra training python $P/datagen/pack_closed_loop.py --accepted $P/cl/pool500 \
  --repo-id local/pusht_cl_verified500 --truncate-cov 0.95 --ep-len 9 --max-pairs 500 \
  --oracle-npz $P/diag/filter_roc_pool500.npz --overwrite
echo '=== PACK DONE ==='

# ---------- 5. aggregate + pin normalization stats to the real-only set ----------
uv run --extra training python -c "
from pathlib import Path; from lerobot.datasets.aggregate import aggregate_datasets
import shutil, json
H=Path.home()/'.cache/huggingface/lerobot'
real=json.load(open(H/'local/pusht_imgonly/meta/stats.json'))
for aug,out_id in (('local/pusht_cl_untrimmed500','local/pusht_plus_untrimmed500'),
                   ('local/pusht_cl_trimmed500','local/pusht_plus_trimmed500'),
                   ('local/pusht_cl_verified500','local/pusht_plus_verified500')):
    out=H/out_id
    if out.exists(): shutil.rmtree(out)
    aggregate_datasets(repo_ids=['local/pusht_imgonly',aug], aggr_repo_id=out_id,
        roots=[H/'local/pusht_imgonly', H/aug], aggr_root=out)
    sp=out/'meta/stats.json'; d=json.load(open(sp))
    for k in ('observation.image','action'): d[k]=real[k]
    json.dump(d, open(sp,'w')); print(out_id,'ready')
"
echo '=== AGGREGATE DONE ==='

# ---------- 6. train + eval all three ----------
for arm in untrimmed500 trimmed500 verified500; do
  rm -rf outputs/train/cl_aug/${arm}_d8_s1000
  bash $P/scripts/finetune_cl.sh $arm 1000 20000 8   # ep-len 9 recipe -> drop 8
  bash $P/scripts/proto_eval_pusht.sh outputs/train/cl_aug/${arm}_d8_s1000/checkpoints/last/pretrained_model \
    outputs/eval/cl_aug/${arm} 200 100000
  echo "=== ${arm} DONE ==="
done
echo '=== DOSE500 ALL DONE ==='
