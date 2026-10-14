#!/usr/bin/env bash
# Method 1 offline check: 8 NanoWM seeds per last-mile candidate (critic/method1_multiseed.py), DIAGNOSTIC ONLY.
# Called by run_method1.sh after the main result (never fatal there). Each sigma is split in 2 parts, one per GPU.
# Resumable: a part whose ms_sigma<X>_p<k>.npz exists is skipped.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=examples/nanowm_pusht_aug
PY="uv run --extra training python"
read -r -a GPU_LIST <<< "${GPUS:-6 7}"
O=$HOME/datastorage/nanowm_pusht_aug/lastmile/method1
mkdir -p "$O/multiseed" "$O/logs"
part() {  # <sigma> <part> <gpu>
  [ -f "$O/multiseed/ms_sigma$1_p$2.npz" ] && { echo "skip sigma $1 part $2"; return 0; }
  CUDA_VISIBLE_DEVICES=$3 $PY $P/critic/method1_multiseed.py run --out "$O" --sigma "$1" --part "$2" --nparts 2
}
for SIG in 0.0 0.1 0.25; do
  echo "$(date '+%F %T') sigma $SIG"
  ( part "$SIG" 0 "${GPU_LIST[0]}" > "$O/logs/ms_${SIG}_p0.log" 2>&1 ) & p0=$!
  ( part "$SIG" 1 "${GPU_LIST[1 % ${#GPU_LIST[@]}]}" > "$O/logs/ms_${SIG}_p1.log" 2>&1 ) & p1=$!
  fail=0; wait "$p0" || fail=1; wait "$p1" || fail=1
  [ $fail = 0 ] || { echo "sigma $SIG failed"; exit 1; }
done
$PY $P/critic/method1_multiseed.py analyze --out "$O"
