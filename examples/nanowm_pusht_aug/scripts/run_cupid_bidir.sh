#!/usr/bin/env bash
# CUPID bidirectional test on verified464 (setting: add-one-in), end to end. The shared config is
# the only input; it points at the two direction configs (LOO / AOI) that each run their own
# record/featurize/score against the SAME pool, so every arm downstream differs only in WHICH pairs
# were kept, never in the estimator.
#
#   bash examples/nanowm_pusht_aug/scripts/run_cupid_bidir.sh <shared_config.yaml> [stage ...]
#   stages (default: all, in this order):
#     sidecar    write pair_index.json for the pool          -> ~/.cache/.../pusht_cl_verified464/meta/
#     record     Stage 2 (per direction) -- test rollouts    -> <dir out>/test_rollouts.npz
#     featurize  Stage 3 (per direction) -- projected grads  -> <dir out>/features/
#     replicate  fail-fast -- featurize again with seeds + 1 -> <dir out>/features_r1/
#     score      Stage 3 (per direction) -- pair scores      -> <dir out>/pair_scores.csv
#     gate       rank-stability gate per direction            -> <bidir out>/gate.json (never aborts)
#     build      Stage 4 -- 5 arms (loo/aoi cupid + 3 random) -> local/pusht_plus_verified464_..._k{k}
#     finetune   re-fine-tune every arm from the base policy  -> outputs/train/cl_aug/<tag>_d<drop>_s<seed>
#     eval       n-episode eval of every arm + the references -> <bidir out>/eval/<tag>/
#     report     success table + paired comparisons + gate    -> <bidir out>/report.md
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG="${1:?usage: run_cupid_bidir.sh <shared_config.yaml> [stage ...]}"
shift
STAGES="${*:-sidecar record featurize replicate score gate build finetune eval report}"
A=examples/nanowm_pusht_aug/attribution
E=examples/nanowm_pusht_aug/eval
S=examples/nanowm_pusht_aug/scripts
D=examples/nanowm_pusht_aug/datagen
PY="uv run --extra training python"

y() { uv run python "$A/config_io.py" "$CFG" "$1"; }   # top-level: shared config
y2() { uv run python "$A/config_io.py" "$1" "$2"; }     # a specific (direction) config
read_dict() {  # read_dict <dotted key of a dict in the shared config> -> "name value" lines, robustly
  A="$A" CFG="$CFG" KEY="$1" uv run python - <<'PYEOF'
import os, sys

sys.path.insert(0, os.environ["A"])
from config_io import load_config

cfg = load_config(os.environ["CFG"], out_dir_copy=False)
node = cfg
for part in os.environ["KEY"].split("."):
    node = node[part]
for name, val in node.items():
    print(name, val if val is not None else "")
PYEOF
}

OUT=$(y bidir.out_dir)
DROP=$(y drop_n_last_frames)
SEED=$(y finetune.seed)
STEPS=$(y finetune.steps)

declare -A DIRCFG
while read -r name path; do [ -n "$name" ] && DIRCFG[$name]="$path"; done < <(read_dict bidir.directions)
declare -A REFDIRS
while read -r name path; do [ -n "$name" ] && REFDIRS[$name]="$path"; done < <(read_dict bidir.refs)

arms() { $PY $A/build_bidir_datasets.py --config "$CFG" --list; }
ckpt_of() { echo "outputs/train/cl_aug/$1_d${DROP}_s${SEED}/checkpoints/last/pretrained_model"; }

for stage in $STAGES; do
  echo "=== $stage $(date +%H:%M) ==="
  case "$stage" in
    sidecar)
      $PY $D/pack_closed_loop.py --accepted examples/nanowm_pusht_aug/cl/pool500 \
        --repo-id local/pusht_cl_verified464 --truncate-cov 0.95 --ep-len 9 --max-pairs 464 \
        --oracle-npz examples/nanowm_pusht_aug/diag/filter_roc_pool500.npz --index-only ;;
    record)
      for name in "${!DIRCFG[@]}"; do
        dcfg="${DIRCFG[$name]}"
        dir_out=$(y2 "$dcfg" out_dir)
        if [ -f "$dir_out/test_rollouts.npz" ]; then
          echo "skip $name record (test_rollouts.npz exists -- re-recording would invalidate its features)"
          continue
        fi
        $PY $E/record_policy_rollouts.py --config "$dcfg"
      done ;;
    featurize)
      for name in "${!DIRCFG[@]}"; do $PY $A/featurize.py --config "${DIRCFG[$name]}"; done ;;
    replicate)
      for name in "${!DIRCFG[@]}"; do $PY $A/featurize.py --config "${DIRCFG[$name]}" --replicate 1; done ;;
    score)
      for name in "${!DIRCFG[@]}"; do
        uv run --extra training --with matplotlib python $A/score_pairs.py --config "${DIRCFG[$name]}"
      done ;;
    gate)
      uv run python - "$OUT" $(for name in "${!DIRCFG[@]}"; do echo "$name" "${DIRCFG[$name]}"; done) <<'PYEOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, "examples/nanowm_pusht_aug/attribution")
from config_io import load_config

out_dir, rest = Path(sys.argv[1]), sys.argv[2:]
gate = {}
for name, cfg_path in zip(rest[0::2], rest[1::2], strict=True):
    dcfg = load_config(cfg_path, out_dir_copy=False)
    summary = json.loads((Path(dcfg["out_dir"]) / "score_summary.json").read_text())
    rho = summary.get("spearman_features_vs_features_r1")
    r_avg = None if rho is None else 2 * rho / (1 + rho)
    gate[name] = {"rho": rho, "r_avg": r_avg, "pass": r_avg is not None and r_avg >= 0.8}
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "gate.json").write_text(json.dumps(gate, indent=1))
print(json.dumps(gate, indent=1))
PYEOF
      ;;
    build) $PY $A/build_bidir_datasets.py --config "$CFG" ;;
    finetune)
      for repo in $(arms); do
        tag="${repo#local/}"
        if [ -d "$(ckpt_of "$tag")" ]; then
          echo "skip $tag (checkpoint exists)"
          continue
        fi
        SAVE_FREQ=$(y finetune.save_freq) bash $S/finetune_cl.sh "repo=$repo" "$SEED" "$STEPS" "$DROP"
      done ;;
    eval)
      NEP=$(y eval.n_episodes); ESEED=$(y eval.seed); NA=$(y eval.n_action_steps)
      for repo in $(arms); do
        tag="${repo#local/}"
        [ -f "$OUT/eval/$tag/eval_info.json" ] && { echo "skip $tag (done)"; continue; }
        bash $S/proto_eval_pusht.sh "$(ckpt_of "$tag")" "$OUT/eval/$tag" "$NEP" "$ESEED" "$NA"
      done
      for name in "${!REFDIRS[@]}"; do
        ref="${REFDIRS[$name]}"
        if [ -n "$ref" ]; then
          [ -e "$OUT/eval/$name" ] || { mkdir -p "$OUT/eval"; ln -s "$(realpath "$ref")" "$OUT/eval/$name"; }
          echo "ref $name -> $ref (symlinked)"
        else
          [ -f "$OUT/eval/$name/eval_info.json" ] && { echo "skip ref $name (done)"; continue; }
          bash $S/proto_eval_pusht.sh "$(ckpt_of "$name")" "$OUT/eval/$name" "$NEP" "$ESEED" "$NA"
        fi
      done ;;
    report)
      LOO_TAG=$(arms | sed 's#^local/##' | grep '_loo_cupid_')
      AOI_TAG=$(arms | sed 's#^local/##' | grep '_aoi_cupid_')
      uv run python $A/paired_report.py "$OUT" --random-prefix random_s --ref a_real_only \
        --pair "$LOO_TAG:$AOI_TAG"
      {
        echo
        echo "## gate (rank-stability, per direction)"
        echo
        echo '```json'
        cat "$OUT/gate.json"
        echo '```'
      } >> "$OUT/report.md" ;;
    *) echo "unknown stage: $stage"; exit 1 ;;
  esac
done
echo "=== run_cupid_bidir.sh done $(date +%H:%M) ==="
