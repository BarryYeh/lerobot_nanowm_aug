#!/usr/bin/env python
"""early_anchors step 2b: pick the new base checkpoint and write the shared configs/base_policy_result.yaml.

Reads <out_dir>/base_eval/<step>/success_steps.npz (written by train_base.sh eval). The picked checkpoint is the one
whose success rate lies in base_train.target_success and is closest to the band's midpoint; with none in the band the
script lists the rates and exits non-zero (the pick is then the user's call).

    uv run python research/nanowm_aug/proj/pusht/early_anchors/pick_base.py --wandb-url <run url>
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from ea_common import CFG, RES

from nwaug.attribution.config_io import load_config

DEST = RES / "configs" / "base_policy_result.yaml"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-url", default=None)
    ap.add_argument(
        "--pick", default=None, help="force a checkpoint step (e.g. 040000) instead of the band rule"
    )
    a = ap.parse_args()
    cfg = load_config(CFG, out_dir_copy=False)
    bt, ev = cfg["base_train"], cfg["eval"]
    eval_dir = Path(cfg["out_dir"]) / "base_eval"
    rows = {}
    for d in sorted(eval_dir.iterdir()):
        f = d / "success_steps.npz"
        if not f.exists():
            continue
        z = np.load(f)
        assert int(z["seed"][0]) == ev["seed"], f"{f}: episodes do not start at eval.seed"
        k, n = int(z["success"].sum()), len(z["success"])
        lo, hi = wilson(k, n)
        ok = z["success_step"][z["success"]]
        rows[d.name] = {
            "success_rate": round(k / n, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_episodes": n,
            "median_success_step": int(np.median(ok)) if len(ok) else None,
        }
    steps = list(rows)
    for s in sorted(rows):
        print(f"  {s:>8}: {rows[s]['success_rate']:.3f}  CI {rows[s]['ci95']}")
    lo_t, hi_t = bt["target_success"]
    if a.pick:
        pick = a.pick
    else:
        band = [s for s in steps if lo_t <= rows[s]["success_rate"] <= hi_t]
        if not band:
            raise SystemExit(f"no checkpoint in the target band {bt['target_success']}; pass --pick <step>")
        pick = min(band, key=lambda s: abs(rows[s]["success_rate"] - (lo_t + hi_t) / 2))
    ckpt = str(Path(bt["output_dir"]).expanduser() / "checkpoints" / pick / "pretrained_model")
    out = {
        "_comment": (
            "Shared base-policy results (written by proj/pusht/early_anchors/pick_base.py; do not hand-edit). "
            "Every success rate uses the `eval` protocol below on the SAME episodes, so rows are comparable."
        ),
        "generated": date.today().isoformat(),
        "eval_protocol": {k: ev[k] for k in ("n_episodes", "seed", "n_action_steps", "batch_size")},
        "success_threshold": cfg["pusht"]["success_threshold"],
        "early_anchors_base": {
            "picked_step": pick,
            "ckpt": ckpt,
            "pick_rule": f"success in {bt['target_success']}, closest to its midpoint"
            if not a.pick
            else "manual",
            "train": {
                "recipe": bt["recipe"],
                "steps": bt["steps"],
                "save_freq": bt["save_freq"],
                "seed": bt["seed"],
                "dataset": cfg["datasets"]["real"],
                "output_dir": bt["output_dir"],
                "wandb": a.wandb_url,
            },
            "checkpoints": {s: rows[s] for s in sorted(steps)},
        },
    }
    DEST.write_text(yaml.safe_dump(out, sort_keys=False, width=120))
    print(f"picked {pick} ({rows[pick]['success_rate']:.3f}); wrote {DEST}")


if __name__ == "__main__":
    main()
