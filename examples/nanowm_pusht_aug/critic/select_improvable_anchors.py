#!/usr/bin/env python
"""SAILOR expand S2: keep the candidate anchors where the base policy still has room to improve.

The base policy is rolled `expand.screen.reps` times from every candidate anchor (`seeds_candidates.npz`,
written by make_reset_anchors.py) with exactly the label continuation's noise (`label.cont_perturb`, prefix mode,
whole rollout), for `label.B + LEAD` steps from the candidate's CoG state (the real anchor is this state plus the
10-step imagination lead-in, so the screen is a proxy; S4 re-measures the anchor itself at r=0). Per candidate:
p0 = success rate, t0 = median success step over the successes. A candidate is kept unless the base policy is
already reliable AND fast (p0 >= easy_p0 and t0 < easy_t0: nothing to speed up) or nearly hopeless (p0 < p0_min).
The first `expand.n_anchors` kept candidates, in candidate order, become `seeds_stage2.npz` (every per-candidate
array filtered, scalars kept, plus cand_index / screen_p0 / screen_t0).

    screen  --shard i --n-shards N   roll a contiguous slice of candidates -> <out>/screen/part_<i>.npz
    merge                            combine the parts, apply the rule  -> seeds_stage2.npz + screen.json

Torch seeding is `expand.screen.seed` with `batch_start` = the global row index of each batch (row = cand * reps
+ rep), so results do not depend on which GPU ran a shard; the noise rng is seeded per shard.

    uv run --extra training python examples/nanowm_pusht_aug/critic/select_improvable_anchors.py \
        --config configs/sailor_expand.yaml screen --shard 0 --n-shards 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402

BATCH = 100


def keep_mask(p0: np.ndarray, t0: np.ndarray, p0_min: float, easy_p0: float, easy_t0: float) -> np.ndarray:
    """True = the base policy has headroom from this anchor. t0 is NaN when there was no success."""
    p0, t0 = np.asarray(p0, float), np.asarray(t0, float)
    easy = (p0 >= easy_p0) & (np.nan_to_num(t0, nan=np.inf) < easy_t0)
    return (p0 >= p0_min) & ~easy


def summarize(success_step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, reps) success steps (-1 = none) -> p0 (n,), t0 (n,) median over successes (NaN if none)."""
    ok = success_step > 0
    p0 = ok.mean(1)
    t0 = np.array([np.median(s[o]) if o.any() else np.nan for s, o in zip(success_step, ok, strict=True)])
    return p0, t0


def paths(cfg: dict, smoke: bool) -> dict:
    base = Path(cfg["out_dir"]).expanduser() / ("smoke" if smoke else "")
    return {
        "base": base,
        "cand": base / "seeds_candidates.npz",
        "screen": base / "screen",
        "out": base / "seeds_stage2.npz",
        "json": base / "screen.json",
    }


def screen(cfg: dict, smoke: bool, shard: int, n_shards: int) -> None:
    import torch
    from label_imagined_branches import LEAD
    from policy_io import load_policy
    from sim_rollout import rollout_batch

    P = paths(cfg, smoke)
    sc, lab = cfg["expand"]["screen"], cfg["label"]
    reps = int(cfg["expand"]["smoke"]["screen_reps"] if smoke else sc["reps"])
    cog = np.load(P["cand"])["cog_states"].astype(np.float64)
    n = len(cog)
    lo, hi = shard * n // n_shards, (shard + 1) * n // n_shards
    out = P["screen"] / f"part_{shard}.npz"
    if out.exists():
        print(f"skip {out} (exists)")
        return
    cp = lab["cont_perturb"]
    B = int(lab["B"]) + LEAD
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=device)
    rng = np.random.default_rng([int(sc["seed"]) + 1, shard])
    rows = np.repeat(np.arange(lo, hi), reps)  # candidate index of every row, candidate-major
    succ, wall = np.zeros(len(rows), np.int64), np.zeros(len(rows), bool)
    t0 = time.time()
    for b0 in range(0, len(rows), BATCH):
        idx = rows[b0 : b0 + BATCH]
        res = rollout_batch(
            policy,
            pre,
            post,
            cog[idx],
            device=device,
            na=int(lab["na"]),
            episode_steps=B,
            seed=int(sc["seed"]),
            batch_start=lo * reps + b0,
            sigma=np.full(len(idx), float(cp["sigma"])),
            perturb_p=np.full(len(idx), B, np.int64),
            max_z=float(cp["max_z"]),
            noise_hold=int(cp["noise_hold"]),
            noise_rng=rng,
            lockstep224=False,
            frame_cutoff_step=0,
            wall_pen_px=float(lab["wall_pen_px"]),
            perturb_mode="prefix",
        )
        succ[b0 : b0 + len(idx)] = res["success_step"]
        wall[b0 : b0 + len(idx)] = res["wall_artifact"]
        print(f"  shard {shard}: {b0 + len(idx)}/{len(rows)} rollouts ({time.time() - t0:.0f}s)", flush=True)
    P["screen"].mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez(
        tmp,
        cand=np.arange(lo, hi),
        success_step=succ.reshape(hi - lo, reps),
        wall=wall.reshape(hi - lo, reps),
        reps=np.int64(reps),
        B=np.int64(B),
    )
    tmp.rename(out)
    print(f"wrote {out}")


def merge(cfg: dict, smoke: bool, n_shards: int) -> None:
    P = paths(cfg, smoke)
    sc = cfg["expand"]["screen"]
    n_keep = int(cfg["expand"]["smoke"]["n_anchors"] if smoke else cfg["expand"]["n_anchors"])
    parts = [np.load(P["screen"] / f"part_{i}.npz") for i in range(n_shards)]
    cand = np.concatenate([p["cand"] for p in parts])
    ss = np.concatenate([p["success_step"] for p in parts])
    wall = np.concatenate([p["wall"] for p in parts])
    d = np.load(P["cand"], allow_pickle=True)
    n = len(d["cog_states"])
    assert np.array_equal(cand, np.arange(n)), "screen parts do not cover every candidate exactly once"
    p0, t0 = summarize(ss)
    keep = keep_mask(p0, t0, float(sc["p0_min"]), float(sc["easy_p0"]), float(sc["easy_t0"]))
    sel = np.flatnonzero(keep)[:n_keep]
    info = {
        "n_candidates": n,
        "n_headroom": int(keep.sum()),
        "n_kept": len(sel),
        "rule": {k: sc[k] for k in ("p0_min", "easy_p0", "easy_t0")},
        "reps": int(ss.shape[1]),
        "candidates": [
            {
                "cand": int(i),
                "p0": float(p0[i]),
                "t0": None if np.isnan(t0[i]) else float(t0[i]),
                "walls": int(wall[i].sum()),
                "headroom": bool(keep[i]),
                "kept": bool(i in set(sel)),
            }
            for i in range(n)
        ],
    }
    P["json"].write_text(json.dumps(info, indent=1))
    if len(sel) < n_keep:
        raise SystemExit(
            f"only {len(sel)} candidates with headroom (< {n_keep}); see {P['json']} -- raise "
            f"stage2.anchors.n rather than loosening the rule"
        )
    out = {}
    for k in d.files:
        v = d[k]
        out[k] = v[sel] if v.ndim and len(v) == n else v
    out.update(
        cand_index=sel.astype(np.int64),
        screen_p0=p0[sel].astype(np.float32),
        screen_t0=t0[sel].astype(np.float32),
    )
    tmp = P["out"].with_name(P["out"].stem + ".partial.npz")
    np.savez(tmp, **out)
    tmp.rename(P["out"])
    print(
        f"{len(sel)}/{n} candidates kept ({int(keep.sum())} with headroom); p0 kept mean {p0[sel].mean():.3f}, "
        f"t0 kept median {np.nanmedian(t0[sel]):.0f} -> {P['out']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--n-shards", type=int, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--n-shards", type=int, required=True)
    a = ap.parse_args()
    cfg = load_config(a.config, out_dir_copy=False)
    if a.cmd == "screen":
        screen(cfg, a.smoke, a.shard, a.n_shards)
    else:
        merge(cfg, a.smoke, a.n_shards)


if __name__ == "__main__":
    main()
