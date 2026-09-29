#!/usr/bin/env python
"""E2 -- with the block pose FROZEN, which agent positions can the base policy recover from?

E1 showed that at some top-wall-band states every mode of the diffusion policy fails (s0: 0/9 modes,
block never leaves the band). The reposition idea says the fault is where the AGENT stands, not the
block pose. This maps that directly: teleport the agent to each cell of a grid, leave the block
exactly where it is, hand control to the unmodified base policy, and record whether it succeeds.

  a high-value region exists  -> repositioning is the thing to learn, and this map IS the score
                                 function for it
  the map is flat and low     -> that block pose is unrecoverable regardless of the agent, and
                                 dreaming has nothing to find

The agent is teleported (velocity zeroed, policy queue reset), so the map is an UPPER BOUND: it says
where it is good to be, not whether the agent can get there without disturbing the block. Cells whose
straight line from the episode's own start crosses the block are flagged `needs_detour`.

Cells are skipped when the agent circle would overlap the block (< --clear-px from its outline) or
leave the walls. Two rollouts per cell is deliberate: the readable quantity is the SIZE and SHAPE of
the high-value region after 3x3 smoothing (effective n = 18), not any single cell.

    P=examples/nanowm_pusht_aug
    uv run --extra training --with matplotlib python $P/eval/agent_value_map.py \
        --starts-npz $P/cl/e1_modes10/e1_rollouts.npz --start-ids 0 1 --grid-step 25 --rollouts 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_EVAL = Path(__file__).resolve().parent
_PROJ = _EVAL.parent
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_EVAL))
from ood_failure_map import (  # noqa: E402
    T_OUTLINE,
    sample_topband_starts,
    set_state,
    sim_state,
)

BASE_CKPT = Path("outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model")
AGENT_BOX = (22.0, 489.0)


def block_polygon(state: np.ndarray) -> np.ndarray:
    """(5,) CoG state -> (8,2) T outline in world pixels."""
    c, s = np.cos(state[4]), np.sin(state[4])
    R = np.array([[c, -s], [s, c]])
    return T_OUTLINE @ R.T + state[2:4]


def dist_to_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(N,2) points -> distance to the polygon boundary (0 inside is not distinguished; the caller
    only needs 'far enough away')."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None] - proj, axis=-1).min(1)


def segment_hits_polygon(p0: np.ndarray, pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Does the straight line p0 -> pts[i] cross the polygon? (N,) bool."""
    a, b = poly, np.roll(poly, -1, axis=0)

    def cross(o, u, v):
        return (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - (u[..., 1] - o[..., 1]) * (
            v[..., 0] - o[..., 0]
        )

    p1 = pts[:, None]
    d1 = cross(a[None], b[None], np.broadcast_to(p0, p1.shape))
    d2 = cross(a[None], b[None], p1)
    d3 = cross(np.broadcast_to(p0, p1.shape), p1, a[None])
    d4 = cross(np.broadcast_to(p0, p1.shape), p1, b[None])
    return (((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))).any(1)


def main() -> None:
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=BASE_CKPT)
    p.add_argument("--starts-npz", type=Path, default=None, help="reuse starts from an E1 run")
    p.add_argument("--start-ids", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-starts", type=int, default=10, help="if no --starts-npz: resample this many")
    p.add_argument("--seed", type=int, default=400_000)
    p.add_argument("--grid-step", type=float, default=25.0)
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument(
        "--clear-px", type=float, default=35.0, help="min agent-centre distance to the block outline"
    )
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=250, help="envs stepped in lock-step at once")
    p.add_argument("--out", type=Path, default=Path("examples/nanowm_pusht_aug/cl/e2_value_map"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    args.out.mkdir(parents=True, exist_ok=True)

    if args.starts_npz:
        all_starts = np.load(args.starts_npz)["starts"]
    else:
        all_starts = sample_topband_starts(args.n_starts, args.seed)
    starts = all_starts[args.start_ids]

    g = np.arange(50.0, 450.0 + 1e-9, args.grid_step)
    gx, gy = np.meshgrid(g, g, indexing="xy")
    cells_all = np.stack([gx.ravel(), gy.ravel()], 1)

    jobs, meta = [], []
    for i, st in zip(args.start_ids, starts, strict=True):
        poly = block_polygon(st)
        ok = (
            (dist_to_polygon(cells_all, poly) >= args.clear_px)
            & (cells_all >= AGENT_BOX[0]).all(1)
            & (cells_all <= AGENT_BOX[1]).all(1)
        )
        cells = cells_all[ok]
        detour = segment_hits_polygon(st[:2], cells, poly)
        meta.append(
            {"start_id": int(i), "state": st.tolist(), "cells": cells, "valid_mask": ok, "detour": detour}
        )
        for c in cells:
            for r in range(args.rollouts):
                jobs.append((len(meta) - 1, c, r))
    print(f"starts={args.start_ids} grid={len(g)}x{len(g)} step={args.grid_step}px "
          f"valid cells={[len(m['cells']) for m in meta]} rollouts={args.rollouts} episodes={len(jobs)} "
          f"(~{len(jobs) * 0.7 / 60:.0f} min at 0.7 s/ep)")  # fmt: skip

    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )
    torch.manual_seed(args.seed)

    success = np.zeros(len(jobs), bool)
    max_cov = np.zeros(len(jobs))
    end_y = np.zeros(len(jobs))
    for b0 in range(0, len(jobs), args.batch):
        chunk = jobs[b0 : b0 + args.batch]
        envs = [gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", observation_width=96,
                         observation_height=96, max_episode_steps=args.max_steps + 10) for _ in chunk]  # fmt: skip
        obs = []
        for e, (mi, cell, _r) in zip(envs, chunk, strict=True):
            e.reset(seed=b0 + len(obs))
            st = np.asarray(meta[mi]["state"], float).copy()
            st[:2] = cell  # teleport the agent, block pose untouched
            obs.append(set_state(e.unwrapped, st)["pixels"])
        policy.reset()
        done = np.zeros(len(chunk), bool)
        for _t in range(args.max_steps):
            img = torch.from_numpy(np.stack(obs)).permute(0, 3, 1, 2).float().div(255.0).to(dev)
            with torch.no_grad():
                a = (
                    post(policy.select_action(pre({"observation.image": img})))
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            for j, e in enumerate(envs):
                if done[j]:
                    continue
                o, _r, term, trunc, info = e.step(a[j])
                obs[j] = o["pixels"]
                max_cov[b0 + j] = max(max_cov[b0 + j], float(info.get("coverage", 0.0)))
                if term or trunc:
                    done[j] = True
                    success[b0 + j] = bool(term)
            if done.all():
                break
        for j, e in enumerate(envs):
            end_y[b0 + j] = sim_state(e.unwrapped)[3]
            e.close()
        print(
            f"  {min(b0 + args.batch, len(jobs))}/{len(jobs)} episodes  running success {success[: b0 + len(chunk)].mean():.2f}",
            flush=True,
        )

    res = {"ckpt": str(args.ckpt), "grid_step": args.grid_step, "rollouts": args.rollouts,
           "clear_px": args.clear_px, "per_start": []}  # fmt: skip
    job_mi = np.array([j[0] for j in jobs])
    for mi, m in enumerate(meta):
        sel = job_mi == mi
        v = success[sel].reshape(len(m["cells"]), args.rollouts).mean(1)
        esc = (end_y[sel].reshape(len(m["cells"]), args.rollouts) >= 145).mean(1)
        # 3x3 smoothing on the full grid, so "is there a REGION" is read at effective n = 9*rollouts
        full = np.full(len(cells_all), np.nan)
        full[m["valid_mask"]] = v
        grid = full.reshape(len(g), len(g))
        sm = np.full_like(grid, np.nan)
        for a_ in range(len(g)):
            for b_ in range(len(g)):
                w = grid[max(0, a_ - 1) : a_ + 2, max(0, b_ - 1) : b_ + 2]
                if np.isfinite(w).any():
                    sm[a_, b_] = np.nanmean(w)
        row = {"start_id": m["start_id"], "state": m["state"], "n_cells": int(len(m["cells"])),
               "mean_value": float(np.nanmean(v)), "max_value": float(np.nanmax(v)),
               "frac_cells_ge_0.8": float(np.nanmean(v >= 0.8)),
               "frac_smoothed_ge_0.8": float(np.nanmean(sm >= 0.8)),
               "escape_rate_mean": float(np.nanmean(esc)),
               "best_cell": m["cells"][int(np.nanargmax(v))].tolist(),
               "frac_needs_detour_among_ge_0.8": float(m["detour"][v >= 0.8].mean()) if (v >= 0.8).any() else None}  # fmt: skip
        res["per_start"].append(row)
        np.savez_compressed(args.out / f"map_s{m['start_id']}.npz", cells=m["cells"], value=v, escape=esc,
                            detour=m["detour"], state=np.asarray(m["state"]), grid=grid, smoothed=sm, axis=g)  # fmt: skip
        print(f"\nstart s{m['start_id']}: cells={row['n_cells']}  mean V={row['mean_value']:.2f}  max V={row['max_value']:.2f}  "
              f"cells>=0.8: {row['frac_cells_ge_0.8']:.2f} (smoothed {row['frac_smoothed_ge_0.8']:.2f})  "
              f"escape {row['escape_rate_mean']:.2f}  best cell {row['best_cell']}  "
              f"needs-detour among good cells {row['frac_needs_detour_among_ge_0.8']}")  # fmt: skip
    (args.out / "e2_summary.json").write_text(json.dumps(res, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing -> no figure")
        return
    fig, axes = plt.subplots(1, 2 * len(meta), figsize=(7 * len(meta), 6))
    for k, m in enumerate(meta):
        d = np.load(args.out / f"map_s{m['start_id']}.npz")
        st = np.asarray(m["state"], float)
        poly = np.vstack([block_polygon(st), block_polygon(st)[:1]])
        for ax, arr, name in (
            (axes[2 * k], d["grid"], "raw"),
            (axes[2 * k + 1], d["smoothed"], "3x3 smoothed"),
        ):
            im = ax.imshow(
                arr, origin="upper", extent=[g[0], g[-1], g[-1], g[0]], vmin=0, vmax=1, cmap="viridis"
            )
            ax.plot(poly[:, 0], poly[:, 1], "w-", lw=2)
            ax.plot(*st[:2], "r*", ms=14, label="original agent")
            ax.plot(256, 256, "g+", ms=12, label="goal centre")
            ax.set_title(f"s{m['start_id']} {name}: P(success | agent here)")
            ax.set_xlim(40, 460)
            ax.set_ylim(460, 40)
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[2 * k].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "e2_value_maps.png", dpi=120)
    print(f"wrote {args.out / 'e2_value_maps.png'}")


if __name__ == "__main__":
    main()
