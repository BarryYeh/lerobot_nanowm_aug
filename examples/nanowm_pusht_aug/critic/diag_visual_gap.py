#!/usr/bin/env python
"""Diagnostic: how much does the base policy's action change when a REAL frame is swapped for the imagined one?

For each branch (the replicated ones in selected/ plus --n-random other stage-2 branches), the branch is replayed in a
224px and a 96px gym_pusht env in lock step (place_seed -> lead-in -> exec_abs_targets, asserted against
agent_state_hist at every round). At every round k = 1..h_max we have, for the SAME simulator moment:
  img   the NanoWM frame of round k (224) -> 96 with the pipeline's own resize (closed_loop_rollout._to96)
  real  the 224px sim render -> the same resize (what arm B's pairs use)
  r96   the native 96px sim render (what the policy sees when deployed)
The base policy samples a chunk for each with the same diffusion noise (paired: identical torch seed and batch split).
Distances = mean over the first n_action_steps actions of the L2 gap (px):
  noise   real vs real with a different noise draw   (the policy's own sampling spread: the yardstick)
  resize  r96 vs real, same noise                     (224->96 resize path vs native render)
  img     img vs real, same noise                     (the imagined-frame gap)
Output <out>/visual_gap.json / .md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
sys.path.insert(0, str(_CRITIC.parent / "eval"))
sys.path.insert(0, str(_CRITIC.parent))
from closed_loop_rollout import _to96, sample_chunks  # noqa: E402
from config_io import load_config  # noqa: E402
from label_imagined_branches import FI, TOL, agent_pv, place_seed  # noqa: E402
from policy_io import load_policy  # noqa: E402

BS = 256


def replay_frames(z: dict) -> tuple[np.ndarray, np.ndarray]:
    """(h_max,224,224,3) and (h_max,96,96,3) real frames after every round."""
    h = len(z["frames"])
    envs = [
        gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels",
            observation_width=w,
            observation_height=w,
            max_episode_steps=10_000,
        )
        for w in (224, 96)
    ]
    for e in envs:
        place_seed(e, int(z["seed_index"]), z["seed_state"])
        for a in z["lead_acts"]:
            e.step(a)
    f224, f96 = [], []
    for t, a in enumerate(z["exec_abs_targets"][: h * FI]):
        for e in envs:
            e.step(a)
        if (t + 1) % FI == 0:
            k = (t + 1) // FI
            for e in envs:
                err = float(np.abs(agent_pv(e.unwrapped) - z["agent_state_hist"][k]).max())
                assert err <= TOL, f"round {k}: agent state off by {err:.2e}"
            f224.append(np.asarray(envs[0].unwrapped.get_obs(), np.uint8))
            f96.append(np.asarray(envs[1].unwrapped.get_obs(), np.uint8))
    for e in envs:
        e.close()
    return np.stack(f224), np.stack(f96)


def sample_all(policy, pre, post, frames96: np.ndarray, seed: int, dev: str) -> np.ndarray:
    out = []
    for i in range(0, len(frames96), BS):
        torch.manual_seed(seed * 100_003 + i)
        out.append(sample_chunks(policy, pre, post, frames96[i : i + BS], 1, dev))
    return np.concatenate(out)


def gap(a: np.ndarray, b: np.ndarray, na: int) -> np.ndarray:
    return np.linalg.norm(a[:, :na] - b[:, :na], axis=-1).mean(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sailor_expand.yaml"))
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--noise-draws", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, out_dir_copy=False)
    base = Path(cfg["out_dir"]).expanduser()
    out = a.out or base / "diag_visual"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    na = int(cfg["label"]["na"])
    sel = sorted((base / "selected").glob("*.npz"))
    sel_real = {p.resolve() for p in sel}
    pool = [p for p in sorted((base / "stage2").glob("sigma*/branches/*.npz")) if p.resolve() not in sel_real]
    rnd = [
        pool[i] for i in np.random.default_rng(0).choice(len(pool), min(a.n_random, len(pool)), replace=False)
    ]
    files = [(p, "selected") for p in sel] + [(p, "random") for p in rnd]
    img, real, r96, group, rnd_k = [], [], [], [], []
    for p, g in files:
        z = np.load(p)
        f224, f96 = replay_frames(z)
        for k in range(len(f224)):
            img.append(_to96(z["frames"][k]))
            real.append(_to96(f224[k]))
            r96.append(f96[k])
            group.append(g)
            rnd_k.append(k + 1)
    img, real, r96 = np.stack(img), np.stack(real), np.stack(r96)
    group, rnd_k = np.array(group), np.array(rnd_k)
    print(f"{len(files)} branches ({len(sel)} selected, {len(rnd)} random) -> {len(img)} frame triplets")
    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=dev)
    d = {"noise": [], "resize": [], "img": []}
    for s in range(a.noise_draws):
        c_real = sample_all(policy, pre, post, real, s, dev)
        c_real2 = sample_all(policy, pre, post, real, s + 1000, dev)
        c_img = sample_all(policy, pre, post, img, s, dev)
        c_r96 = sample_all(policy, pre, post, r96, s, dev)
        d["noise"].append(gap(c_real, c_real2, na))
        d["resize"].append(gap(c_r96, c_real, na))
        d["img"].append(gap(c_img, c_real, na))
    d = {k: np.stack(v).mean(0) for k, v in d.items()}  # per frame, averaged over noise draws

    def stats(mask):
        return {
            k: {"median": float(np.median(v[mask])), "mean": float(v[mask].mean())} for k, v in d.items()
        } | {
            "n": int(mask.sum()),
            "img_over_noise_median": float(np.median(d["img"][mask]) / np.median(d["noise"][mask])),
        }

    rep = {
        "n_actions": na,
        "noise_draws": a.noise_draws,
        "all": stats(np.ones(len(img), bool)),
        "selected": stats(group == "selected"),
        "random": stats(group == "random"),
        "by_round": {int(k): stats(rnd_k == k) for k in sorted(set(rnd_k.tolist()))},
    }
    (out / "visual_gap.json").write_text(json.dumps(rep, indent=1))
    lines = [
        f"# Base-policy action gap, imagined vs real frame ({len(img)} frame triplets, first {na} actions, px)",
        "",
        "| set | n | noise (same frame, other noise) | resize (native 96 vs 224->96) | img (imagined vs real) | img / noise |",
        "|---|---|---|---|---|---|",
    ]
    for name, st in [("all", rep["all"]), ("selected", rep["selected"]), ("random", rep["random"])] + [
        (f"round {k}", v) for k, v in rep["by_round"].items()
    ]:
        lines.append(
            f"| {name} | {st['n']} | {st['noise']['median']:.1f} | {st['resize']['median']:.1f} | "
            f"{st['img']['median']:.1f} | {st['img_over_noise_median']:.2f} |"
        )
    (out / "visual_gap.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
