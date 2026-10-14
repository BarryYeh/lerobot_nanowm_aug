#!/usr/bin/env python
"""Band starts for CUPID add-one-in (stage 2): three disjoint sets of top-wall-band states.

sample_topband_starts() returns CoG-based states, but closed_loop_rollout.py places the block by its
BODY origin (angle first, then block.position). Mixing the two conventions is the +45 px reset bug
from the OOD study, so every seed file carries both:
  states      [agent x, agent y, body x, body y, theta]  -> closed_loop_rollout.py --seeds
  cog_states  [agent x, agent y, CoG x, CoG y, theta]    -> set_state() users (test rollouts, band eval)
Each state is placed exactly the way closed_loop_rollout.py does it and must land its CoG within
0.5 px of the sampled one, or the script fails.

    uv run --extra training python examples/nanowm_pusht_aug/eval/make_topband_seeds.py \
        --config configs/cupid_select_topband.yaml

Default: the three CUPID sets (topband.pool / .test / .eval -> seeds_{pool,test,eval}.npz).
With `--section a.b --out-name f.npz` it writes ONE set instead, from the config node cfg[a][b]
({n, seed, min_wall_gap}), e.g. the value-model stage-2 anchors:
    ... --config configs/sailor_v1_reset.yaml --section stage2.anchors --out-name seeds_stage2.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pymunk

_EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL))
sys.path.insert(0, str(_EVAL.parent))
sys.path.insert(0, str(_EVAL.parent / "attribution"))
from config_io import load_config  # noqa: E402
from nwaug.pusht_state import place_body, save_seeds  # noqa: E402
from ood_failure_map import sample_topband_starts, sim_state  # noqa: E402


def cog_to_body(cog_states: np.ndarray, cog_local: pymunk.Vec2d) -> np.ndarray:
    body = cog_states.copy()
    for i, s in enumerate(cog_states):
        off = cog_local.rotated(float(s[4]))
        body[i, 2:4] = s[2] - off.x, s[3] - off.y
    return body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--section",
        default=None,
        help="dotted config path of ONE {n, seed, min_wall_gap} node; default = the CUPID topband sets",
    )
    p.add_argument(
        "--out-name", default=None, help="output file name under out_dir (required with --section)"
    )
    args = p.parse_args()
    if (args.section is None) != (args.out_name is None):
        raise SystemExit("--section and --out-name must be given together")
    cfg = load_config(args.config)
    out = Path(cfg["out_dir"])
    band = cfg["pusht"]["topband_cog_y"]
    if args.section is None:
        tb = cfg["topband"]
        jobs = [
            (name, tb[name], tb["min_wall_gap"], f"seeds_{name}.npz") for name in ("pool", "test", "eval")
        ]
    else:
        node = cfg
        for key in args.section.split("."):
            node = node[key]
        jobs = [(args.section, node, node["min_wall_gap"], args.out_name)]

    env = gym.make("gym_pusht/PushT-v0")
    env.reset(seed=0)
    u = env.unwrapped
    cog_local = pymunk.Vec2d(*u.block.center_of_gravity)

    for name, spec, min_gap, fname in jobs:
        cog = sample_topband_starts(spec["n"], spec["seed"], min_gap)
        body = cog_to_body(cog, cog_local)
        err = np.zeros(len(cog))
        for i, st in enumerate(body):
            env.reset(seed=0)
            place_body(u, st)
            err[i] = np.hypot(*(sim_state(u)[2:4] - cog[i, 2:4]))
        if err.max() > 0.5:
            raise SystemExit(f"{name}: body/CoG round trip off by {err.max():.2f} px -- convention mismatch")
        assert ((cog[:, 3] >= band[0]) & (cog[:, 3] < band[1])).all()
        save_seeds(out / fname, body=body, cog=cog)
        print(
            f"{name}: {len(cog)} starts (seed {spec['seed']}), CoG y [{cog[:, 3].min():.1f}, "
            f"{cog[:, 3].max():.1f}], placement err max {err.max():.3f} px -> {out / fname}"
        )


if __name__ == "__main__":
    main()
