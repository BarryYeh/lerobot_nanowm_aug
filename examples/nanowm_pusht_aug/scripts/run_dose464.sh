#!/usr/bin/env bash
# Equal-dose comparison at N=464 pairs for all three arms.
# 464 is the ceiling: Verified yields only 464 candidate pairs from the 2,290-branch pool
# (Untrimmed 7,894 / Trimmed 1,501 / Verified 464). Holding all three at the common maximum
# isolates LABEL QUALITY from DOSE, which was the confound in the earlier 161-vs-503 result.
set -euo pipefail
# repo root -- derived from this script's own location (examples/nanowm_pusht_aug/eval/)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=examples/nanowm_pusht_aug
N=464

uv run --extra training python $P/datagen/pack_closed_loop.py --accepted $P/cl/pool500 \
  --repo-id local/pusht_cl_untrimmed464 --pair-cov 0.95 --ep-len 9 --max-pairs $N --overwrite
uv run --extra training python $P/datagen/pack_closed_loop.py --accepted $P/cl/pool500 \
  --repo-id local/pusht_cl_trimmed464 --truncate-cov 0.95 --ep-len 9 --max-pairs $N --overwrite
uv run --extra training python $P/datagen/pack_closed_loop.py --accepted $P/cl/pool500 \
  --repo-id local/pusht_cl_verified464 --truncate-cov 0.95 --ep-len 9 --max-pairs $N \
  --oracle-npz $P/diag/filter_roc_pool500.npz --overwrite
echo '=== PACK464 DONE ==='

uv run --extra training python -c "
from pathlib import Path; from lerobot.datasets.aggregate import aggregate_datasets
import shutil, json
H=Path.home()/'.cache/huggingface/lerobot'
real=json.load(open(H/'local/pusht_imgonly/meta/stats.json'))
for aug,out_id in (('local/pusht_cl_untrimmed464','local/pusht_plus_untrimmed464'),
                   ('local/pusht_cl_trimmed464','local/pusht_plus_trimmed464'),
                   ('local/pusht_cl_verified464','local/pusht_plus_verified464')):
    out=H/out_id
    if out.exists(): shutil.rmtree(out)
    aggregate_datasets(repo_ids=['local/pusht_imgonly',aug], aggr_repo_id=out_id,
        roots=[H/'local/pusht_imgonly', H/aug], aggr_root=out)
    sp=out/'meta/stats.json'; d=json.load(open(sp))
    for k in ('observation.image','action'): d[k]=real[k]
    json.dump(d, open(sp,'w')); print(out_id,'ready')
"
echo '=== AGGREGATE464 DONE ==='

for arm in untrimmed trimmed verified; do
  rm -rf outputs/train/cl_aug/${arm}464_d8_s1000
  bash $P/scripts/finetune_cl.sh $arm 1000 20000 8   # ep-len 9 recipe -> drop 8
  bash $P/scripts/proto_eval_pusht.sh outputs/train/cl_aug/${arm}464_d8_s1000/checkpoints/last/pretrained_model \
    outputs/eval/cl_aug/${arm}464 200 100000
  echo "=== ${arm}464 DONE ==="
done
echo '=== DOSE464 ALL DONE ==='
