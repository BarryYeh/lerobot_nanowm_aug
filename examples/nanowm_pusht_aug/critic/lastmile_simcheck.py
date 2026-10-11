#!/usr/bin/env python
"""Last-mile experiment, gate 2 (DIAGNOSTIC ONLY -- never selects or labels training data).

Replays every candidate of every accepted start exactly in the simulator (label_imagined_branches.replay_row: seed
placement, lead-in asserted against anchor_state, the branch's executed chunk) and records whether the simulator
reaches success (coverage >= 0.95) within the chunk. Reports, over accepted starts, the true success rate of the W
and R picks, the share of starts where ANY candidate truly succeeds (the ceiling for any selector), and the
precision / recall of "imagined score >= thr" as a success call. Gate 2: rate(W) - rate(R) >= --min-gap.
Writes <sel>/simcheck.json; exit code 1 when the gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent))
from label_imagined_branches import FI, LEAD, replay_row  # noqa: E402


def sim_success(path: Path, stats: dict) -> bool:
    z = np.load(path)
    br = {k: z[k] for k in z.files if k not in ("frames", "anchor_frame", "latents")}
    h = len(z["coverage"])
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", max_episode_steps=LEAD + h * FI + 10)
    rp = replay_row(env, br, h, path.name, stats)
    env.close()
    return bool(rp["terminated"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="dir holding sigma*/branches")
    ap.add_argument("--sel", type=Path, required=True, help="select_lastmile.py --out dir")
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--min-gap", type=float, default=0.10)
    a = ap.parse_args()
    sel = json.loads((a.sel / "selection.json").read_text())
    acc = [r for r in sel["rows"] if r["accepted"]]
    stats = {"anchor_checks": 0, "max_anchor_err": 0.0, "agent_checks": 0, "max_agent_err": 0.0}
    w_ok, r_ok, any_ok, calls = [], [], [], []
    for r in acc:
        cands = sorted(a.root.glob(f"sigma*/branches/seed{r['start']:04d}_cand*.npz"))
        res = {}
        for f in cands:
            name = f"{f.parent.parent.name}_{f.name}"
            res[name] = sim_success(f, stats)
            calls.append((float(np.max(np.load(f)["coverage"])) >= a.thr, res[name]))
        w_ok.append(res[r["W"]])
        r_ok.append(res[r["R"]])
        any_ok.append(any(res.values()))
    pred = np.array([c[0] for c in calls], bool)
    true = np.array([c[1] for c in calls], bool)
    tp = int((pred & true).sum())
    mean = lambda x: float(np.mean(x)) if len(x) else 0.0  # noqa: E731
    out = {
        "n_accepted": len(acc),
        "W_true_success": mean(w_ok),
        "R_true_success": mean(r_ok),
        "any_candidate_true_success": mean(any_ok),
        "imagined_call_precision": tp / max(1, int(pred.sum())),
        "imagined_call_recall": tp / max(1, int(true.sum())),
        "n_candidates": len(calls),
        "replay_checks": stats,
    }
    out["gap_W_minus_R"] = out["W_true_success"] - out["R_true_success"]
    out["gate2_pass"] = bool(out["gap_W_minus_R"] >= a.min_gap)
    (a.sel / "simcheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if not out["gate2_pass"]:
        print(f"GATE 2 FAILED: W - R = {out['gap_W_minus_R']:+.3f} < {a.min_gap}")
        sys.exit(1)


if __name__ == "__main__":
    main()
