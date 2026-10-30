#!/usr/bin/env python
"""imagined_replay step 3: round-1 NanoWM frames F̂(s) for the collected success trajectories, sliding 1 sim step.

    F̂(s) <- NanoWM( F(s-15), F(s-10), F(s-5), a[s-15 : s+5] )      s = t_start+1 .. T

Every window is one row of the NanoWM `closed-loop` worker run for a single round (`imagined_rollout.py`,
nanowm venv), so arbitrary (trajectory, s) windows batch together. The context is always real, nothing compounds.

Action window. NanoWM takes RELATIVE per-step displacements: rel[t] = a[t] - agent_pos[t], with agent_pos[t] the
agent position BEFORE step t. Here the simulator is available, so agent_pos is read from an exact replay of the
episode (env.reset(seed) + all executed actions), not from the free-PD estimate the closed loop has to use. The
replay must end in success at exactly T (asserted). `ac.rel_to_nanowm` then normalizes and reshapes to (4, 10).

The last action group a[s : s+5] never reaches the network: NanoWM.forward shifts the action embeddings
(`cat([zeros, emb[:, :-1]])`, third_party/nano-world-model/src/models/nanowm.py), so frame j is conditioned on the
group that ENTERS it and the 4th group is dropped. For s > T-5 that group runs past T; it is padded with the last
action (any value gives the same frame). The collected trajectories store real frames only up to T-5, so the real
F(T-4 .. T) are re-rendered from the same replay (checked bit for bit against the stored F(T-5)).

  run      --select terminal : only s = T-5 per trajectory (the fidelity filter input)
           --select all      : every s = t_start+1 .. T-5
           --select tail     : s = T-4 .. T (the steps the first pass skipped), real F(s) saved alongside
           Sharded with --part/--parts (one process per GPU); chunks of --chunk windows, one worker per chunk;
           a finished chunk leaves <out>/<select>/part<p>_c<i>.npz (resumable).
  summary  Per-window diagnostics (coverage gap, block CoG error from the image pose). --select terminal: the old
           filter |cov(F̂(T-5)) - cov(F(T-5))| <= --thr. --select tail: the filter F̂(T) meets the success criterion,
           image coverage >= --success-cov (0.95, the simulator's success threshold) -> fidelity_filter_T.json.

    CUDA_VISIBLE_DEVICES=6 uv run --extra training python .../replay_imagine.py run --collect <c> --out <o> \
        --select terminal --part 0 --parts 2
    uv run --extra training python .../replay_imagine.py summary --out <o> --select terminal --thr 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
from ir_common import CTX, FI, NEED, NF, NSS, RES, WM_BATCH, WM_SEED, H, load_starts

from nwaug.wm.closed_loop_rollout import NANOWM_PY, _save_atomic, _wait
from nwaug_envs.pusht import action_convert as ac
from nwaug_envs.pusht.coverage import coverage_from_frame

WORKER = RES / "nwaug" / "wm" / "imagined_rollout.py"


def agent_track(traj: dict, starts: dict, render_tail: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Exact agent position before every step t in [lo, T): replay the whole episode in the simulator.

    render_tail: also return the real 224px frames F(T-5 .. T), rendered exactly as collect_success.py did
    (a 224px `pixels` env, `get_obs()` after t steps); F(T-5) must equal the stored frame bit for bit."""
    si, lo, T = int(traj["start_id"]), int(traj["action_steps"][0]), int(traj["T"])
    acts = np.concatenate([starts["prefix"][si, :lo], traj["actions"]], axis=0)  # a[0 : T]
    assert len(acts) == T, (len(acts), T)
    kw = {"obs_type": "pixels", "observation_width": 224, "observation_height": 224} if render_tail else {}
    env = gym.make("gym_pusht/PushT-v0", max_episode_steps=T + 10, **({"obs_type": "state"} | kw))
    env.reset(seed=int(traj["env_seed"]))
    u = env.unwrapped
    pos = np.zeros((T - lo, 2))
    tail = np.zeros((FI + 1, 224, 224, 3), np.uint8) if render_tail else None  # F(T-5 .. T)
    for t in range(T + 1):
        if render_tail and t >= T - FI:
            tail[t - (T - FI)] = np.asarray(u.get_obs())
        if t == T:
            break
        if t >= lo:
            pos[t - lo] = np.asarray(u.agent.position)
        _o, _r, term, _tr, info = env.step(acts[t])
        assert not term or t == T - 1, f"{traj['name']}: replay ended at {t + 1} < T={T}"
    assert term and bool(info["is_success"]), f"{traj['name']}: replay did not succeed at T={T}"
    env.close()
    if render_tail:
        stored = traj["frames224"][T - FI - int(traj["frame_steps"][0])]
        assert np.array_equal(tail[0], stored), (
            f"{traj['name']}: re-rendered F(T-5) differs from the stored frame"
        )
    return pos, tail  # pos[i] = agent position before step lo + i


def window_list(collect: Path, select: str) -> list[tuple[str, int]]:
    rows = []
    for p in sorted((collect / "traj").glob("start*.npz")):
        z = np.load(p)
        T, ts = int(z["T"]), int(z["t_start"])
        ss = {
            "terminal": [T - FI],
            "all": list(range(ts + 1, T - FI + 1)),
            "tail": list(range(T - FI + 1, T + 1)),
        }[select]
        rows += [(p.name, s) for s in ss]
    return rows


def build_chunk(
    collect: Path, rows: list[tuple[str, int]], starts: dict, cache: dict, tail: bool = False
) -> dict:
    ctx = np.zeros((len(rows), H, 224, 224, 3), np.uint8)
    real = np.zeros((len(rows), 224, 224, 3), np.uint8)
    win = np.zeros((len(rows), NF, FI * 2), np.float32)
    for i, (name, s) in enumerate(rows):
        if name not in cache:
            z = dict(np.load(collect / "traj" / name))
            z["name"] = name
            z["agent_pos"], z["tail"] = agent_track(z, starts, render_tail=tail)
            cache.clear()  # rows are grouped by trajectory: keep one in memory
            cache[name] = z
        z = cache[name]
        f0, a0 = int(z["frame_steps"][0]), int(z["action_steps"][0])
        for j, t in enumerate(range(s - CTX, s, FI)):  # F(s-15), F(s-10), F(s-5)
            ctx[i, j] = z["frames224"][t - f0]
        T = int(z["T"])
        real[i] = z["frames224"][s - f0] if s <= T - FI else z["tail"][s - (T - FI)]
        seg = slice(s - CTX - a0, s - CTX - a0 + NEED)
        a, pos = z["actions"][seg], z["agent_pos"][seg]
        if len(a) < NEED:  # s > T-5: the dropped 4th group runs past T, pad it (see the module docstring)
            assert s > T - FI and len(a) >= NEED - FI, (name, s, len(a))
            pad = NEED - len(a)
            a = np.concatenate([a, np.repeat(a[-1:], pad, axis=0)])
            pos = np.concatenate([pos, np.repeat(pos[-1:], pad, axis=0)])
        win[i] = ac.rel_to_nanowm(a - pos)
    return {"ctx": ctx, "real": real, "win": win}


def run_worker(work: Path, ctx: np.ndarray, win: np.ndarray, seed: int, a) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*"):
        f.unlink()
    _save_atomic(work / "seeds_expanded.npz", frames_224=ctx)
    _save_atomic(work / "chunks_0.npz", win=win, live=np.ones(len(win), bool))
    with open(work / "wm_worker.log", "w") as log:
        proc = subprocess.Popen(
            [
                str(NANOWM_PY),
                str(WORKER),
                "closed-loop",
                "--work",
                str(work),
                "--rounds",
                "1",
                "--history",
                str(H),
                "--num-sampling-steps",
                str(a.num_sampling_steps),
                "--wm-batch",
                str(a.wm_batch),
                "--seed",
                str(seed),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        _wait(work / "frames_1.npz", timeout=7200.0, proc=proc)
        proc.wait(timeout=120)
    fr = np.load(work / "frames_1.npz")
    return {"frames": fr["frames"], "cov": fr["cov"]}


def frame_stats(img: np.ndarray, real: np.ndarray) -> dict:
    ci = [coverage_from_frame(f) for f in img]
    cr = [coverage_from_frame(f) for f in real]
    pi = np.array([c["pose_img"] for c in ci])
    pr = np.array([c["pose_img"] for c in cr])
    return {
        "cov_img": np.array([c["coverage_cal"] for c in ci], np.float32),
        "cov_real": np.array([c["coverage_cal"] for c in cr], np.float32),
        "cog_err_px": np.linalg.norm(pi[:, :2] - pr[:, :2], axis=1).astype(np.float32),  # 224px image
    }


def cmd_run(a) -> None:
    starts = load_starts(a.collect / "starts.npz")
    rows = window_list(a.collect, a.select)
    mine = rows[a.part :: a.parts] if a.select == "terminal" else None
    if mine is None:  # keep each trajectory's windows in one part (the agent replay runs once per trajectory)
        names = sorted({n for n, _ in rows})
        keep = set(names[a.part :: a.parts])
        mine = [r for r in rows if r[0] in keep]
    if a.limit:
        mine = mine[: a.limit]
    out = a.out / a.select
    out.mkdir(parents=True, exist_ok=True)
    chunks = [mine[i : i + a.chunk] for i in range(0, len(mine), a.chunk)]
    print(
        f"[part {a.part}] {len(mine)}/{len(rows)} windows, {len(chunks)} chunks of <= {a.chunk}", flush=True
    )
    cache: dict = {}
    t0 = time.time()
    for ci, rows_c in enumerate(chunks):
        dest = out / f"part{a.part}_c{ci:03d}.npz"
        if dest.exists():
            continue
        b = build_chunk(a.collect, rows_c, starts, cache, tail=a.select == "tail")
        seed = WM_SEED + 1000 * a.part + ci
        res = run_worker(a.out / f"work_{a.select}_p{a.part}", b["ctx"], b["win"], seed, a)
        st = frame_stats(res["frames"], b["real"])
        _save_atomic(
            dest,
            traj=np.array([n for n, _ in rows_c]),
            s=np.array([s for _, s in rows_c], np.int64),
            frames_img=res["frames"],
            cov_worker=res["cov"],
            wm_seed=np.int64(seed),
            num_sampling_steps=np.int64(a.num_sampling_steps),
            **({"frames_real": b["real"]} if a.select == "tail" else {}),
            **st,
        )
        print(
            f"  [part {a.part}] chunk {ci + 1}/{len(chunks)}: {len(rows_c)} windows, "
            f"CoG err median {np.median(st['cog_err_px']):.2f} px, "
            f"|cov gap| median {np.median(np.abs(st['cov_img'] - st['cov_real'])):.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )


def cmd_summary(a) -> None:
    keys = ("traj", "s", "cov_img", "cov_real", "cog_err_px")
    parts = [np.load(p) for p in sorted((a.out / a.select).glob("part*_c*.npz"))]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    gap = np.abs(d["cov_img"] - d["cov_real"])
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [10, 25, 50, 75, 90])]  # noqa: E731
    summ = {
        "select": a.select,
        "windows": len(gap),
        "cog_err_px_p10_25_50_75_90": q(d["cog_err_px"][np.isfinite(d["cog_err_px"])]),
        "cog_err_nan": int((~np.isfinite(d["cog_err_px"])).sum()),
        "abs_cov_gap_p10_25_50_75_90": q(gap),
        "cov_real_p10_25_50_75_90": q(d["cov_real"]),
        "cov_img_p10_25_50_75_90": q(d["cov_img"]),
    }
    if a.select == "terminal":
        summ["pass_rate_by_thr"] = {
            f"{t:.2f}": float((gap <= t).mean()) for t in (0.01, 0.02, 0.03, 0.05, 0.1)
        }
        if a.thr is not None:
            ok = gap <= a.thr
            summ["thr"] = a.thr
            summ["pass"] = int(ok.sum())
            filt = {str(n): bool(o) for n, o in zip(d["traj"], ok, strict=True)}
            (a.out / "fidelity_filter.json").write_text(
                json.dumps(
                    {"thr": a.thr, "rule": "|cov(F̂(T-5)) - cov(F(T-5))| <= thr", "pass": filt}, indent=1
                )
            )
    if a.select == "tail":
        T_of = {str(n): int(np.load(a.collect / "traj" / str(n))["T"]) for n in set(d["traj"])}
        at_t = np.array([s == T_of[str(n)] for n, s in zip(d["traj"], d["s"], strict=True)])
        ok = d["cov_img"][at_t] >= a.success_cov
        summ["success_cov"] = a.success_cov
        summ["trajectories"] = int(at_t.sum())
        summ["pass_F_hat_T"] = int(ok.sum())
        summ["real_F_T_image_cov_pass"] = int((d["cov_real"][at_t] >= a.success_cov).sum())
        filt = {str(n): bool(o) for n, o in zip(d["traj"][at_t], ok, strict=True)}
        (a.out / "fidelity_filter_T.json").write_text(
            json.dumps(
                {"success_cov": a.success_cov, "rule": "coverage_cal(F̂(T)) >= success_cov", "pass": filt},
                indent=1,
            )
        )
    (a.out / f"summary_{a.select}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--collect", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--select", choices=["terminal", "all", "tail"], required=True)
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--parts", type=int, default=1)
    r.add_argument("--chunk", type=int, default=1024)
    r.add_argument("--num-sampling-steps", type=int, default=NSS)
    r.add_argument("--wm-batch", type=int, default=WM_BATCH)
    r.add_argument("--limit", type=int, default=0, help="smoke test: only the first N windows of this part")
    m = sub.add_parser("summary")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--select", choices=["terminal", "all", "tail"], required=True)
    m.add_argument("--thr", type=float, default=None)
    m.add_argument("--collect", type=Path, default=None, help="--select tail: the collect dir (reads each T)")
    m.add_argument("--success-cov", type=float, default=0.95)
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
