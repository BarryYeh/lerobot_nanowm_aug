#!/usr/bin/env python
"""Stage-2 anchors for the SAILOR v1 value model: states drawn from reset rollouts of the base policy.

For source episode i (`env.reset(seed = node.seed + i)` -> CoG start state via `sim_state`) run the
base policy in `rollout_batch` with EXACTLY stage 1's perturbation settings (D2: `node.perturb`,
default = `stage2.anchors.perturb`; `mode: prefix`, `switch: all|random`), `lockstep224=False`,
policy torch seed `node.seed`. Each source episode yields AT MOST ONE anchor: a time t drawn uniformly
from `t_choices` (`labels.pick_anchor`), rejecting episodes with a wall artifact, an earlier/equal
success (`success_step <= t`) or, with `stage1.query_keep: after_switch`, a query the prefix rule drops
(`labels.keep_query`; with `query_keep: all` -- the noise-everywhere backup -- no query is dropped).
Source episodes are drawn in batches (1.5x the need first) until `n` anchors exist; the first `n` in
source order are kept. The anchor state is the CoG state at step t (grid index
t // 5), converted to body-origin (`make_topband_seeds.cog_to_body`) and verified to round-trip
through `nwaug.pusht_state.place_body` within 0.5 px.

Output npz (`nwaug.pusht_state.save_seeds`): states (body), cog_states, t, perturb_p, src_seed (scalar,
the namespace base), src_episode (reset(seed=src_seed+src_episode)), src_sigma, anchor_set
("train" for stage2.anchors, "test" for test_anchors), perturb_mode, provenance_json.

    uv run --extra training python examples/nanowm_pusht_aug/critic/make_reset_anchors.py \
        --config configs/sailor_v1_reset.yaml --section stage2.anchors --out <dir>/seeds_stage2.npz [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pymunk
import torch

_CRITIC = Path(__file__).resolve().parent
_PROJ = _CRITIC.parent
for _p in (_CRITIC, _PROJ / "eval", _PROJ / "attribution", _PROJ):
    sys.path.insert(0, str(_p))
from config_io import load_config  # noqa: E402
from datagen.provenance import provenance_entry  # noqa: E402
from make_topband_seeds import cog_to_body  # noqa: E402
from nwaug.pusht_state import place_body, save_seeds  # noqa: E402
from ood_failure_map import sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

from critic import labels  # noqa: E402
from critic.labels import pick_anchor  # noqa: E402,F401  (re-exported for tests)

OVERDRAW = 1.5  # source episodes drawn per needed anchor, first round
MAX_ROUNDS_FACTOR = 4  # give up (hard error) after this multiple of the first-round draw


def _node(cfg: dict, section: str) -> dict:
    node = cfg
    for key in section.split("."):
        node = node[key]
    return node


def resolve_settings(cfg: dict, section: str) -> dict:
    """Section node + fallbacks: `t_choices` / `perturb` default to `stage2.anchors`' (test anchors reuse
    them), and a node without its own `perturb` gets a copy of stage2.anchors' with `seed = node.seed`."""
    node = _node(cfg, section)
    ref = cfg["stage2"]["anchors"]
    perturb = dict(node["perturb"]) if "perturb" in node else {**ref["perturb"], "seed": node["seed"]}
    return {
        "seed": int(node["seed"]),
        "n": int(node["n"]),
        "t_choices": [int(t) for t in node.get("t_choices", ref["t_choices"])],
        "perturb": perturb,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--section", choices=["stage2.anchors", "test_anchors"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="n from smoke.*; --out must live under a smoke/ dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    st = resolve_settings(cfg, args.section)
    pr, t_choices, seed = st["perturb"], st["t_choices"], st["seed"]
    anchor_set = "train" if args.section == "stage2.anchors" else "test"
    n = st["n"]
    if args.smoke:
        assert "smoke" in args.out.resolve().parts, f"--smoke would write outside a smoke/ dir: {args.out}"
        n = int(cfg["smoke"]["stage2_anchors" if anchor_set == "train" else "test_anchors"])
    query_keep = cfg["stage1"]["query_keep"]  # D2: anchors obey exactly stage 1's query rule
    assert query_keep in labels.QUERY_KEEPS, query_keep
    mode = pr.get("mode", "point")
    assert mode == "prefix", f"anchor source rollouts need perturb.mode=prefix (D2), got {mode!r}"

    s1 = cfg["stage1"]
    na, wall_pen = cfg["label"]["na"], cfg["label"]["wall_pen_px"]
    ep_steps_p = int(s1["episode_steps"])  # value of perturb_p under `switch: all`
    roll_steps = max(t_choices)  # success before t is all we need to know; shorter rollout, same policy RNG
    n_first = math.ceil(OVERDRAW * n)
    n_max = MAX_ROUNDS_FACTOR * n_first
    batch = min(int(s1["batch"]), max(2, n_first)) if args.smoke else int(s1["batch"])
    sigma_all, p_all = labels.draw_perturbations(pr, n_max, ep_steps_p)  # per-source, batch independent
    noise_rng = np.random.default_rng(pr["seed"] + 1)
    t_rng = np.random.default_rng(seed + 2)  # seed / seed+1 are the perturbation draws / action noise
    print(
        f"{args.section}: want {n} anchors (set={anchor_set}), seed {seed}, batch {batch}, perturb "
        f"mode={mode} switch={pr.get('switch', 'random')} sigma={pr['sigmas']} -> {args.out}"
    )

    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=device)
    reset_env = gym.make("gym_pusht/PushT-v0")
    got: list[dict] = []
    n_src, n_rej = 0, {"wall": 0, "success_or_prefix": 0}
    t0 = time.time()
    while len(got) < n:
        if n_src >= n_max:
            raise SystemExit(f"only {len(got)}/{n} anchors after {n_src} source episodes")
        nb = min(batch, n_max - n_src)
        idx = np.arange(n_src, n_src + nb)
        starts = []
        for i in idx:
            reset_env.reset(seed=seed + int(i))
            starts.append(sim_state(reset_env.unwrapped))
        out = rollout_batch(
            policy,
            pre,
            post,
            np.asarray(starts, np.float64),
            device=device,
            na=na,
            episode_steps=roll_steps,
            seed=seed,
            batch_start=n_src,
            sigma=sigma_all[idx],
            perturb_p=p_all[idx],
            max_z=pr["max_z"],
            noise_hold=pr["noise_hold"],
            noise_rng=noise_rng,
            lockstep224=False,
            frame_grid_stride=5,
            frame_cutoff_step=roll_steps,
            wall_pen_px=wall_pen,
            perturb_mode=mode,
        )
        for j, i in enumerate(idx):
            if len(got) >= n:
                break
            t = pick_anchor(
                out["success_step"][j],
                out["wall_artifact"][j],
                t_choices,
                p_all[i],
                sigma_all[i],
                t_rng,
                query_keep=query_keep,
            )
            if t is None:
                n_rej["wall" if out["wall_artifact"][j] else "success_or_prefix"] += 1
                continue
            gi = t // 5
            assert int(out["grid_steps"][gi]) == t, (out["grid_steps"][gi], t)
            got.append(
                {
                    "cog": out["grid_cog"][j, gi].copy(),
                    "t": t,
                    "p": int(p_all[i]),
                    "sigma": float(sigma_all[i]),
                    "ep": int(i),
                }
            )
        n_src += nb
        print(f"  source episodes {n_src}: {len(got)}/{n} anchors  ({time.time() - t0:.0f}s)", flush=True)
    reset_env.close()

    cog = np.stack([g["cog"] for g in got])
    # --- body-origin conversion + 0.5 px place_body round trip (same check as make_topband_seeds) ---
    env = gym.make("gym_pusht/PushT-v0")
    env.reset(seed=0)
    u = env.unwrapped
    body = cog_to_body(cog, pymunk.Vec2d(*u.block.center_of_gravity))
    err = np.zeros(len(cog))
    for i, s5 in enumerate(body):
        env.reset(seed=0)
        place_body(u, s5)
        err[i] = np.hypot(*(sim_state(u)[2:4] - cog[i, 2:4]))
    env.close()
    if err.max() > 0.5:
        raise SystemExit(
            f"{args.section}: body/CoG round trip off by {err.max():.2f} px -- convention mismatch"
        )

    t_arr = np.array([g["t"] for g in got], np.int64)
    save_seeds(
        args.out,
        body=body,
        cog=cog,
        t=t_arr,
        perturb_p=np.array([g["p"] for g in got], np.int64),
        src_seed=np.int64(seed),
        src_episode=np.array([g["ep"] for g in got], np.int64),
        src_sigma=np.array([g["sigma"] for g in got], np.float32),
        anchor_set=anchor_set,
        perturb_mode=mode,
        provenance_json=json.dumps(provenance_entry(args, stage="anchors", section=args.section)),
    )
    hist = {int(t): int((t_arr == t).sum()) for t in sorted(set(t_choices))}
    print(
        f"{args.section}: {len(got)} anchors from {n_src} source episodes (rejected {n_rej}), "
        f"t histogram {hist}, placement err max {err.max():.3f} px -> {args.out}"
    )


if __name__ == "__main__":
    main()
