#!/usr/bin/env python
"""imagined_replay on the last-mile W set: replace each W pair's REAL anchor frame by the round-1 NanoWM frame of the
anchor itself, keep the W label, pack exactly like last-mile W (Img-W arm; Real-W = the existing last-mile W arm).

The anchor problem. A last-mile anchor is reached by placing the body state at lead-in step 0 (agent and block
velocities 0) and running a 10-step policy lead-in; frames exist at steps 0, 5, 10 (anchor = 10). The sliding
window wants F̂(10) <- NanoWM(F(-5), F(0), F(5), a[-5:15]), and step -5 does not exist. Because everything is at
rest at step 0, five HOLD steps (target = the agent's own position) before it leave the state unchanged, so
"rest for 5 steps, then the lead-in" is a real trajectory: F(-5) = F(0), rel actions of those 5 steps are 0.
`imagine` checks this in the simulator per anchor (state after the holds == state at step 0, and the held run
reaches the saved anchor state and frame), then builds the window

    context  F(0), F(0), F(5)             (real, re-rendered; sha1 == the saved lead-in ctx_sha1)
    actions  [hold x5 | lead[0:10] | W exec[0:5]] - exact agent positions (simulator) -> ac.rel_to_nanowm

and runs one NanoWM round per anchor (`replay_imagine.run_worker`), then measures F̂(10) vs the real anchor.

  imagine  -> <out>/imagined.npz (+ fidelity in <out>/fidelity.json)
  pack     -> copies of the W branch files with anchor_frame := F̂(anchor), duplicated like last-mile W, packed with
              the last-mile W arguments (pack_closed_loop.py --ep-len 9 --pair-cov 0.0 --include-anchor), then
              `local/pusht_irw_imgw_plus_real` (build_real_plus_pairs, stats pinned to the real data)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ir_common import FI, LM, NSS, REAL_REPO, RES, WM_BATCH, WM_SEED, lastmile_env, sha1, w_anchors
from replay_imagine import frame_stats, run_worker

from nwaug_envs.pusht import action_convert as ac
from nwaug_envs.pusht.state import read_body

HOLD, LEAD = 5, 10
W_WM_SEED = WM_SEED + 2000  # the imagined_replay_wm block, unused part index 2
DUP = 7  # last-mile W: ceil(1200 / 179)


def rebuild(i: int, seed_state, lead_acts, exec_tgts, hold: bool) -> dict:
    """Re-run lead-in seed i (optionally after HOLD rest steps); frames every FI steps of the lead-in, body states,
    and the agent position before every step through the first FI W actions."""
    env = lastmile_env(i, seed_state)
    u = env.unwrapped
    body0 = read_body(u)
    held = []
    if hold:
        for _ in range(HOLD):
            env.step(np.asarray(u.agent.position, np.float64))
            held.append(read_body(u))
    frames, bodies, pos = [], [], []
    for t, a in enumerate(np.concatenate([lead_acts, exec_tgts[:FI]], 0)):
        if t % FI == 0 and t <= LEAD:
            frames.append(np.asarray(u.get_obs()))
            bodies.append(read_body(u))
        pos.append(np.asarray(u.agent.position, np.float64))
        env.step(a)
    env.close()
    return {"body0": body0, "held": held, "frames": frames, "bodies": bodies, "pos": np.array(pos)}


def cmd_imagine(a) -> None:
    ctx, win, real, names, checks = [], [], [], [], []
    for f, z, _sig, i, li in w_anchors():
        acts = li["lead_acts"][i]
        assert np.array_equal(acts, z["lead_acts"]), f.name
        exec_tgts = z["exec_abs_targets"]
        base = rebuild(i, li["seed_states"][i], acts, exec_tgts, hold=False)
        held = rebuild(i, li["seed_states"][i], acts, exec_tgts, hold=True)
        ok_ctx = sha1(np.stack(base["frames"])) == str(li["ctx_sha1"][i])
        ok_anchor = np.array_equal(base["bodies"][2], li["anchor_state"][i]) and np.array_equal(
            base["frames"][2], z["anchor_frame"]
        )
        # pymunk contact correction can nudge the block during the holds when the placed agent touches it;
        # --rest-tol (sim px) bounds that, 0 = bit-exact rest
        d_rest = max(float(np.abs(h - held["body0"]).max()) for h in held["held"])
        d_held = float(np.abs(held["bodies"][2] - base["bodies"][2]).max())
        ok_rest = d_rest <= max(a.rest_tol, 1e-9)
        ok_held = d_held <= max(a.rest_tol, 1e-9)
        checks.append(
            {
                "file": f.name,
                "ctx_sha1": ok_ctx,
                "anchor": ok_anchor,
                "rest": ok_rest,
                "held": ok_held,
                "rest_max_change": d_rest,
                "held_anchor_diff": d_held,
            }
        )
        rel = np.concatenate(
            [np.zeros((HOLD, 2)), np.concatenate([acts, exec_tgts[:FI]], 0) - base["pos"]], 0
        )
        ctx.append(np.stack([base["frames"][0], base["frames"][0], base["frames"][1]]))
        win.append(ac.rel_to_nanowm(rel).astype(np.float32))
        real.append(base["frames"][2])
        names.append(f.name)
    a.out.mkdir(parents=True, exist_ok=True)
    bad = {k: [c["file"] for c in checks if not c[k]] for k in ("ctx_sha1", "anchor", "rest", "held")}
    print("simulator checks failed per kind:", {k: len(v) for k, v in bad.items()}, flush=True)
    (a.out / "anchor_checks.json").write_text(json.dumps({"failed": bad, "checks": checks}, indent=1))
    if any(bad.values()):
        raise SystemExit(f"anchor reconstruction failed for some W pairs, see {a.out / 'anchor_checks.json'}")
    res = run_worker(a.out / "work", np.stack(ctx), np.stack(win), W_WM_SEED, a)
    st = frame_stats(res["frames"], np.stack(real))
    # offset check: F̂(10) should be closer to F(10) than to F(5)
    f5 = np.stack([c[2] for c in ctx]).astype(np.float32)
    mae = lambda x, y: np.abs(x.astype(np.float32) - y).mean(axis=(1, 2, 3))  # noqa: E731
    np.savez(a.out / "imagined.npz", names=np.array(names), frames_img=res["frames"], **st)
    gap = np.abs(st["cov_img"] - st["cov_real"])
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [10, 25, 50, 75, 90])]  # noqa: E731
    fid = {
        "n": len(names),
        "wm_seed": W_WM_SEED,
        "rest_tol": a.rest_tol,
        "not_bit_exact_rest": [c["file"] for c in checks if c["rest_max_change"] > 1e-9],
        "cog_err_px_p10_25_50_75_90": q(st["cog_err_px"]),
        "abs_cov_gap_p10_25_50_75_90": q(gap),
        "mae_vs_F10": float(mae(res["frames"], np.stack(real).astype(np.float32)).mean()),
        "mae_vs_F5": float(mae(res["frames"], f5).mean()),
    }
    (a.out / "fidelity.json").write_text(json.dumps(fid, indent=1))
    print(json.dumps(fid, indent=1))


def cmd_pack(a) -> None:
    d = np.load(a.out / "imagined.npz")
    img = dict(zip(d["names"].tolist(), d["frames_img"], strict=True))
    src, dup = a.out / "branches_imgw", a.out / "branches_imgw_dup"
    for p in (src, dup):
        p.mkdir(parents=True, exist_ok=True)
        for f in p.glob("*.npz"):
            f.unlink()
    for f in sorted((LM / "sel" / "W").glob("*.npz")):
        z = dict(np.load(f))
        z["anchor_frame"] = img[f.name]
        np.savez(src / f.name, **z)
        for k in range(DUP):
            (dup / f"dup{k}_{f.name}").symlink_to(src / f.name)
    subprocess.run(
        [sys.executable, str(RES / "nwaug/datagen/pack_closed_loop.py"), "--accepted", str(dup), "--repo-id",
         "local/pusht_irw_imgw", "--ep-len", "9", "--pair-cov", "0.0", "--include-anchor", "--overwrite"],
        check=True,
    )  # fmt: skip
    from nwaug.attribution.build_selected_datasets import build_all_plus_real

    n_pool, n = build_all_plus_real(REAL_REPO, "local/pusht_irw_imgw")
    print(f"local/pusht_irw_imgw_plus_real: {n} episodes ({n_pool} pairs + real)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("imagine")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--num-sampling-steps", type=int, default=NSS)
    i.add_argument("--wm-batch", type=int, default=WM_BATCH)
    i.add_argument(
        "--rest-tol", type=float, default=0.0, help="max state change (sim px / rad) during the holds"
    )
    p = sub.add_parser("pack")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"imagine": cmd_imagine, "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
