#!/usr/bin/env python
"""lerobot-eval with per-episode STEP counts (eval_info.json only stores success / reward).

Wraps `lerobot.scripts.lerobot_eval.rollout` (episode i of batch b still uses seed start_seed + b*batch + i, so
the episodes are the ones `proto_eval_pusht.sh` ran) and records, per episode, the first step at which
`is_success` was True. Takes the same CLI as lerobot-eval; `--output_dir` receives success_steps.npz:

  seed (N,), success (N,) bool, success_step (N,) int32 (env steps taken when success first fired; -1 = none),
  ep_len (N,) int32 (env steps taken when the episode ended: success, or the 300-step limit)

    uv run --extra training python examples/nanowm_pusht_aug/eval/eval_success_steps.py \
        --policy.path=<ckpt> --policy.device=cuda --policy.use_amp=false --policy.n_action_steps=10 \
        --env.type=pusht --eval.n_episodes=1000 --eval.batch_size=50 --eval.use_async_envs=false \
        --seed=100000 --output_dir=<out>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.scripts import lerobot_eval as le

_orig_rollout = le.rollout
_rows: list[tuple[np.ndarray, np.ndarray]] = []  # per batch: (success_step (B,), ep_len (B,))


def _rollout(*args, **kwargs):
    data = _orig_rollout(*args, **kwargs)
    succ = data["success"].bool()  # (B, T)
    done = data["done"].bool()  # (B, T), cumulative
    first_succ = torch.where(succ.any(1), succ.int().argmax(1) + 1, torch.full((succ.shape[0],), -1))
    first_done = torch.where(
        done.any(1), done.int().argmax(1) + 1, torch.full((done.shape[0],), done.shape[1])
    )
    _rows.append((first_succ.numpy().astype(np.int32), first_done.numpy().astype(np.int32)))
    return data


le.rollout = _rollout  # eval_policy resolves `rollout` from the module globals at call time


def _arg(name: str) -> str:
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    raise SystemExit(f"missing --{name}=...")


if __name__ == "__main__":
    out = Path(_arg("output_dir"))
    start, n = int(_arg("seed")), int(_arg("eval.n_episodes"))
    le.main()
    ss = np.concatenate([r[0] for r in _rows])[:n]
    el = np.concatenate([r[1] for r in _rows])[:n]
    np.savez(
        out / "success_steps.npz", seed=start + np.arange(len(ss)), success=ss > 0, success_step=ss, ep_len=el
    )
    print(f"wrote {out / 'success_steps.npz'}: {len(ss)} episodes, {int((ss > 0).sum())} successes")
