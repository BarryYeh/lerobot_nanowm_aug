#!/usr/bin/env python
"""T1 -- in the top-wall band, does NanoWM reproduce the REAL rollout under beta-action injection?

Dreaming only helps if NanoWM, given the actions actually executed, predicts what the simulator
does. The OOD target is the top-wall band (block CoG y < 145, never in training), where recovery
needs wall contact and an agent squeezing between block and wall -- the regime NanoWM is least
likely to have seen. This measures that directly, with the real simulator as ground truth.

Action noise is parameterized in NANOWM SIGMA UNITS (--inject sigma, the default):

    rel = (base_policy_action - agent_pos) + k * sigma_px * z,   z ~ N(0, I) held --noise-hold steps
    rel = rel * min(1, max_z / |rel / sigma_px|)                 keeps the action in-distribution
    action = agent_pos + rel,  clipped to the agent-reachable box

sigma_px = 100 * NANOWM_A_STD ~ 20.8 px/step is the std of the actions NanoWM was TRAINED on, so k
is "how many training sigmas of noise per step" and --max-z caps the total action NanoWM is asked
to condition on. The base policy alone already runs at |z| p99 = 3.3, hence --max-z 3 by default.

The first fidelity sweep used the older blend (--inject beta):
    action = (1 - beta) * base_policy_action + beta * random_action
with random_action uniform over the whole reachable box. It is kept to reproduce cl/wmfid/*, but
it conflates two things: beta=0.25 already injects 2.8 sigma/step (p95 5.0) and beta=1.0 reaches
11.5 sigma, which is far outside NanoWM's action distribution -- the frames then lose the block
entirely (14-20% of rounds) rather than merely blurring.

base_policy_action comes from pusht_imgonly/obs1_s100000 acting on the REAL 96px frame (its own
action queue is untouched). The same noise draw is shared by every level of a start (common random
numbers), so levels are compared on identical noise.

Stages (lerobot venv; `wm` spawns the NanoWM worker in the nanowm conda env):

  real     starts x betas rolled in gym_pusht. Stores the executed stream, the EXACT agent
           position/velocity before each target (so the NanoWM relative action is exact -- no
           pd_step approximation), full state, coverage, and 224px frames every fi steps.
           -> <work>/real.npz

  wm       NanoWM fed the same executed stream, obeying the closed_loop_rollout ALIGNMENT CONTRACT:
           context = real frames at steps 0, fi, 2fi; round r window = stream[r*fi : r*fi+nf*fi];
           round r generates the frame at step (H+r)*fi.
             --mode ar  autoregressive: its own frames become context (what dreaming does)
             --mode tf  teacher-forced: real context every round, 1-step error only
           -> <work>/wm_<mode>.npz

  analyze  per round, imagined vs real frame: block CoG error, block angle error, agent error,
           coverage error (same image estimator on both sides), block-vanished rate, and whether
           the block's net motion is predicted. "copy" baseline = last real context frame (i.e.
           "nothing moves"); NanoWM must beat it to be reflecting the rollout at all.
           -> <work>/fidelity_summary.json, figure, side-by-side videos

    P=examples/nanowm_pusht_aug; W=$P/cl/wmfid/topband
    uv run --extra training python $P/eval/wm_fidelity_ood.py real --work $W --region topband --n-starts 48
    uv run --extra training python $P/eval/wm_fidelity_ood.py wm --work $W --mode ar
    uv run --extra training python $P/eval/wm_fidelity_ood.py wm --work $W --mode tf --tf-rounds 0 5 10 15
    uv run --extra training --with matplotlib python $P/eval/wm_fidelity_ood.py analyze \
        --work $W --ref $P/cl/wmfid/reset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_EVAL = Path(__file__).resolve().parent
_PROJ = _EVAL.parent  # examples/nanowm_pusht_aug
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_EVAL))
from datagen import action_convert as ac  # noqa: E402
from datagen.coverage_from_frame import coverage_from_frame  # noqa: E402
from ood_failure_map import sample_topband_starts, set_state, sim_state  # noqa: E402

NANOWM_PY = Path("~/miniconda3/envs/nanowm/bin/python").expanduser()
BASE_CKPT = Path("outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model")
H, FI, NF = 3, 5, 4  # NanoWM context frames, frame interval (sim steps), frames per DFoT call
AGENT_BOX = (22.0, 489.0)  # agent centre range that keeps a radius-15 agent inside the walls
SIGMA_PX = ac.ACTION_SCALE * ac.NANOWM_A_STD  # ~20.8 px/step: std of NanoWM's training actions
IMG = 224
# T outline in body-local coords, shifted so the CoG (0,45) is the origin; for wall-penetration checks
T_OUTLINE = np.array(
    [(-60, 0), (60, 0), (60, 30), (-60, 30), (-15, 30), (-15, 120), (15, 120), (15, 30)], float
) - (0, 45)


def wall_penetration(states: np.ndarray) -> np.ndarray:
    """(...,5) CoG states -> px the T reaches past the walls' inner faces (7 / 504); <=0 = inside."""
    s = states.reshape(-1, 5)
    c, sn = np.cos(s[:, 4])[:, None], np.sin(s[:, 4])[:, None]
    x = s[:, 2, None] + T_OUTLINE[None, :, 0] * c - T_OUTLINE[None, :, 1] * sn
    y = s[:, 3, None] + T_OUTLINE[None, :, 0] * sn + T_OUTLINE[None, :, 1] * c
    p = np.maximum.reduce([7 - x.min(1), x.max(1) - 504, 7 - y.min(1), y.max(1) - 504])
    return p.reshape(states.shape[:-1])


# ------------------------------------------------------------------------------------------- real
def stage_real(args) -> None:
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dev = args.device
    R = args.rounds
    T = (R - 1) * FI + NF * FI  # stream length the last window needs
    n_frames = H + R  # real frames at steps 0, FI, ..., (H+R-1)*FI

    if args.region == "topband":
        starts = sample_topband_starts(args.n_starts, args.seed)
    else:  # gym_pusht's own reset distribution, as the in-distribution reference
        e = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
        starts = []
        for i in range(args.n_starts):
            e.reset(seed=args.seed + i)
            starts.append(sim_state(e.unwrapped))
        e.close()
        starts = np.asarray(starts)
    sigma_mode = args.inject == "sigma"
    levels = np.asarray(args.noise_sigma if sigma_mode else args.betas, np.float64)
    unit = "k(sigma)" if sigma_mode else "beta"
    hold = args.noise_hold if sigma_mode else args.rand_hold
    S, nb = len(starts), len(levels)
    bs = S * nb
    sid, level_b = np.repeat(np.arange(S), nb), np.tile(levels, S)
    rng = np.random.default_rng(args.seed)
    n_hold = math.ceil(T / hold)
    # sigma: unit-variance draw per start, scaled by each level -> common random numbers
    draw = rng.normal(size=(S, n_hold, 2)) if sigma_mode else rng.uniform(*AGENT_BOX, size=(S, n_hold, 2))
    draw_step = np.repeat(np.repeat(draw, hold, axis=1)[:, :T], nb, axis=0)  # (bs,T,2)
    print(
        f"region={args.region} starts={S} inject={args.inject} {unit}={levels.tolist()} hold={hold} "
        f"max_z={args.max_z} branches={bs} rounds={R} stream={T} steps  (1 sigma = {SIGMA_PX.mean():.1f} px/step)"
    )

    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )

    # Two lock-stepped envs per branch, exactly as closed_loop_rollout: the policy sees gym's native
    # 96px render, NanoWM gets gym's native 224px render (env.render() would draw the action marker).
    def mk(kind: str):
        kw = {"observation_width": 96, "observation_height": 96} if kind == "96" else {}
        if kind == "224":
            kw = {"observation_width": IMG, "observation_height": IMG}
        return gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos" if kind == "96" else "pixels",
            max_episode_steps=10 * T,
            **kw,
        )

    e96, e224 = [mk("96") for _ in range(bs)], [mk("224") for _ in range(bs)]
    obs = []
    for b in range(bs):
        e96[b].reset(seed=b)
        e224[b].reset(seed=b)
        obs.append(set_state(e96[b].unwrapped, starts[sid[b]])["pixels"])
        set_state(e224[b].unwrapped, starts[sid[b]])

    frames = np.zeros((bs, n_frames, IMG, IMG, 3), np.uint8)
    acts = np.zeros((bs, T, 2))
    a_base = np.zeros((bs, T, 2))
    apos = np.zeros((bs, T, 2))
    avel = np.zeros((bs, T, 2))
    states = np.zeros((bs, T + 1, 5))
    cov = np.zeros((bs, T + 1))
    success_step = np.full(bs, -1)
    torch.manual_seed(args.seed)
    policy.reset()
    for t in range(T + 1):
        for b in range(bs):
            u = e96[b].unwrapped
            states[b, t] = sim_state(u)
            cov[b, t] = u._get_coverage()
        if t % FI == 0 and t // FI < n_frames:
            for b in range(bs):
                frames[b, t // FI] = np.asarray(e224[b].unwrapped.get_obs())
        if t == T:
            break
        for b in range(bs):
            apos[b, t] = np.asarray(e96[b].unwrapped.agent.position)
            avel[b, t] = np.asarray(e96[b].unwrapped.agent.velocity)
        img = torch.from_numpy(np.stack(obs)).permute(0, 3, 1, 2).float().div(255.0).to(dev)
        with torch.no_grad():
            a_base[:, t] = post(policy.select_action(pre({"observation.image": img}))).cpu().numpy()
        if sigma_mode:
            rel = (a_base[:, t] - apos[:, t]) + level_b[:, None] * SIGMA_PX * draw_step[:, t]
            z = np.linalg.norm(rel / SIGMA_PX, axis=1)  # total action in NanoWM sigma units
            rel *= np.minimum(1.0, args.max_z / np.maximum(z, 1e-9))[:, None]
            acts[:, t] = np.clip(apos[:, t] + rel, *AGENT_BOX)
        else:
            acts[:, t] = (1.0 - level_b[:, None]) * a_base[:, t] + level_b[:, None] * draw_step[:, t]
        for b in range(bs):
            # keep stepping past success: NanoWM is compared over the full stream
            o, _r, term, _tr, _i = e96[b].step(acts[b, t])
            e224[b].step(acts[b, t])
            obs[b] = o["pixels"]
            if term and success_step[b] < 0:
                success_step[b] = t + 1
        if (t + 1) % 20 == 0:
            print(f"  step {t + 1}/{T}", flush=True)
    drift = max(np.abs(sim_state(e224[b].unwrapped) - states[b, T]).max() for b in range(bs))
    print(f"96px/224px env lock-step drift at end: {drift:.2e} (must be ~0)")
    for e in e96 + e224:
        e.close()

    # exact NanoWM windows: rel = target - agent position before that target, from the simulator
    windows = np.zeros((bs, R, NF, FI * 2), np.float32)
    for b in range(bs):
        for r in range(R):
            sl = slice(r * FI, r * FI + NF * FI)
            windows[b, r] = ac.rel_to_nanowm(acts[b, sl] - apos[b, sl])
    pen = wall_penetration(states).max(1)
    args.work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.work / "real.npz",
        starts=starts, betas=levels, sid=sid, beta=level_b, frames=frames, acts=acts, a_base=a_base,
        a_rand=draw_step, apos=apos, avel=avel, states=states, cov=cov, success_step=success_step,
        wall_pen=pen, windows=windows,
        meta=json.dumps({"region": args.region, "ckpt": str(args.ckpt), "rounds": R, "H": H, "fi": FI, "nf": NF,
                         "inject": args.inject, "unit": unit, "hold": hold, "max_z": args.max_z,
                         "sigma_px": SIGMA_PX.tolist(), "seed": args.seed}),
    )  # fmt: skip
    z_all = np.linalg.norm((acts - apos) / SIGMA_PX, axis=2)
    for be in levels:
        m = level_b == be
        print(
            f"  {unit}={be:.2f}: |z| median {np.median(z_all[m]):.2f} p99 {np.percentile(z_all[m], 99):.2f} sigma  "
            f"success {np.mean(success_step[m] >= 0):.2f}  max cov {cov[m].max(1).mean():.2f}  "
            f"block CoG y {states[m, 0, 3].mean():.0f}->{states[m, -1, 3].mean():.0f}  wall-pen>10px {np.mean(pen[m] > 10):.2f}"
        )
    print(f"wrote {args.work / 'real.npz'}")


# --------------------------------------------------------------------------------------------- wm
def _save_atomic(path: Path, **a) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **a)
    tmp.rename(path)


def run_worker(work: Path, seeds: np.ndarray, windows: np.ndarray, args) -> np.ndarray:
    """seeds (bs,H,224,224,3), windows (bs,rounds,NF,2FI) -> imagined frames (bs,rounds,224,224,3).
    Open loop: every window is known up front, so all chunk files are written before the worker starts."""
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*"):
        f.unlink()
    bs, rounds = windows.shape[:2]
    _save_atomic(work / "seeds_expanded.npz", frames_224=seeds)
    for r in range(rounds):
        _save_atomic(work / f"chunks_{r}.npz", win=windows[:, r], live=np.ones(bs, bool))
    cmd = [str(NANOWM_PY), str(_PROJ / "imagined_rollout.py"), "closed-loop",
           "--work", str(work), "--rounds", str(rounds),
           "--history", str(H), "--num-sampling-steps", str(args.num_sampling_steps), "--wm-batch", str(args.wm_batch)]  # fmt: skip
    t0 = time.time()
    with open(work / "wm_worker.log", "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )  # fmt: skip
        try:
            last = work / f"frames_{rounds}.npz"
            while not last.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"NanoWM worker exited early, see {work / 'wm_worker.log'}")
                time.sleep(2)
            (work / "stop").touch()
            proc.wait(timeout=120)
        finally:
            if proc.poll() is None:
                proc.kill()
    print(f"  worker: {bs} branches x {rounds} rounds in {(time.time() - t0) / 60:.1f} min")
    return np.stack([np.load(work / f"frames_{r + 1}.npz")["frames"] for r in range(rounds)], 1)


def stage_wm(args) -> None:
    d = np.load(args.work / "real.npz")
    frames, windows = d["frames"], d["windows"]
    bs, R = windows.shape[:2]
    if args.mode == "ar":
        imag = run_worker(args.work / "worker_ar", frames[:, :H], windows, args)
        np.savez_compressed(args.work / "wm_ar.npz", imag=imag, rounds=np.arange(R))
    else:
        rs = [r for r in args.tf_rounds if r < R]
        seeds = np.concatenate([frames[:, r : r + H] for r in rs], 0)  # (bs*K,H,...) real context per round
        win = np.concatenate([windows[:, r : r + 1] for r in rs], 0)  # one window each
        imag = run_worker(args.work / "worker_tf", seeds, win, args)[:, 0]
        imag = imag.reshape(len(rs), bs, IMG, IMG, 3).transpose(1, 0, 2, 3, 4)  # (bs,K,...)
        np.savez_compressed(args.work / "wm_tf.npz", imag=imag, rounds=np.asarray(rs))
    print(f"wrote {args.work / f'wm_{args.mode}.npz'}")


# ---------------------------------------------------------------------------------------- analyze
def measure(rgb: np.ndarray) -> np.ndarray:
    """224px frame -> [block cog x, cog y (sim px), theta_sim, agent x, agent y, coverage_cal, block area]."""
    c = coverage_from_frame(rgb)
    s = 512 / rgb.shape[0]
    cx, cy, th = c["pose_img"]
    a = rgb.astype(np.int16)
    m = (a[..., 2] - a[..., 0] > 100) & (a[..., 1] < 150)  # RoyalBlue agent
    ax, ay = (np.nonzero(m)[1].mean() * s, np.nonzero(m)[0].mean() * s) if m.sum() > 5 else (np.nan, np.nan)
    return np.array(
        [cx * s, cy * s, (th + np.pi / 2) % (2 * np.pi), ax, ay, c["coverage_cal"], c["block_area_frac"]]
    )


def errors(pred: np.ndarray, real: np.ndarray) -> dict[str, np.ndarray]:
    ang = np.abs(np.angle(np.exp(1j * (pred[..., 2] - real[..., 2]))))
    vanished = pred[..., 6] < 0.3 * real[..., 6]
    return {
        "block_px": np.hypot(pred[..., 0] - real[..., 0], pred[..., 1] - real[..., 1]),
        "angle_deg": np.degrees(ang),
        "agent_px": np.hypot(pred[..., 3] - real[..., 3], pred[..., 4] - real[..., 4]),
        "cov_abs": np.abs(pred[..., 5] - real[..., 5]),
        "vanished": vanished.astype(float),
    }


def analyze_one(work: Path) -> dict:
    d = np.load(work / "real.npz")
    meta = json.loads(str(d["meta"]))
    frames, beta, states, pen = d["frames"], d["beta"], d["states"], d["wall_pen"]
    ar = np.load(work / "wm_ar.npz")["imag"]
    bs, R = ar.shape[:2]
    real_m = np.array([[measure(frames[b, H + r]) for r in range(R)] for b in range(bs)])  # (bs,R,7)
    imag_m = np.array([[measure(ar[b, r]) for r in range(R)] for b in range(bs)])
    ctx_m = np.array([measure(frames[b, H - 1]) for b in range(bs)])  # copy baseline: last real context
    copy_m = np.repeat(ctx_m[:, None], R, 1)
    # estimator sanity against the simulator's exact CoG on the real frames
    exact = states[:, [(H + r) * FI for r in range(R)], 2:4]
    est_err = np.nanmedian(np.hypot(real_m[..., 0] - exact[..., 0], real_m[..., 1] - exact[..., 1]))

    e_wm, e_cp = errors(imag_m, real_m), errors(copy_m, real_m)
    keep = pen <= 10  # drop the wall-penetration sim artifact
    out = {"region": meta["region"], "unit": meta.get("unit", "beta"), "branches": int(bs), "rounds": int(R),
           "estimator_cog_err_px": float(est_err), "wall_pen_excluded": int((~keep).sum()), "by_beta": {}}  # fmt: skip
    tf_path = work / "wm_tf.npz"
    tf = np.load(tf_path) if tf_path.exists() else None
    for be in np.unique(beta):
        m = (beta == be) & keep
        row = {"n": int(m.sum()), "per_round": {}}
        for k in e_wm:
            row["per_round"][k] = (
                np.nanmedian(e_wm[k][m], 0).round(3).tolist()
                if k != "vanished"
                else np.nanmean(e_wm[k][m], 0).round(3).tolist()
            )
            row["per_round"]["copy_" + k] = (
                np.nanmedian(e_cp[k][m], 0).round(3).tolist() if k != "vanished" else None
            )
        # does the WM predict the block's net motion over the horizon?
        dr = real_m[m, -1, :2] - ctx_m[m, :2]
        di = imag_m[m, -1, :2] - ctx_m[m, :2]
        moved = np.hypot(*dr.T) > 20
        ok = np.isfinite(di).all(1) & np.isfinite(dr).all(1)
        row["net_motion"] = {
            "real_moved_gt20px": float(moved.mean()),
            "disp_err_px_median": float(np.nanmedian(np.hypot(*(di - dr).T))),
            "corr_dy": float(np.corrcoef(dr[ok, 1], di[ok, 1])[0, 1]) if ok.sum() > 3 else None,
            "wm_beats_copy_final_block": float(np.nanmean(e_wm["block_px"][m, -1] < e_cp["block_px"][m, -1])),
        }
        if tf is not None:
            rs = tf["rounds"]
            tf_m = np.array([[measure(tf["imag"][b, i]) for i in range(len(rs))] for b in np.nonzero(m)[0]])
            e_tf = errors(tf_m, real_m[m][:, rs])
            row["teacher_forced"] = {
                "rounds": rs.tolist(),
                **{k: np.nanmedian(v, 0).round(3).tolist() for k, v in e_tf.items() if k != "vanished"},
            }
        out["by_beta"][f"{be:.2f}"] = row
    np.savez_compressed(
        work / "fidelity_measures.npz", real=real_m, imag=imag_m, ctx=ctx_m, beta=beta, keep=keep
    )
    return out


def write_video(work: Path, n_per_beta: int = 2) -> None:
    import cv2
    import imageio.v2 as imageio

    d = np.load(work / "real.npz")
    frames, beta, sid, pen = d["frames"], d["beta"], d["sid"], d["wall_pen"]
    ar = np.load(work / "wm_ar.npz")["imag"]
    R = ar.shape[1]
    betas = np.unique(beta)
    good_starts = [s for s in np.unique(sid) if (pen[sid == s] <= 10).all()][:n_per_beta]
    rows = []
    for be in betas:
        cells = []
        for s in good_starts:
            b = int(np.nonzero((sid == s) & (beta == be))[0][0])
            cells.append((b, be, s))
        rows.append(cells)
    video = []
    for t in range(H + R):  # context frames first (real in both panels), then imagined rounds
        grid = []
        for cells in rows:
            line = []
            for b, be, s in cells:
                real = frames[b, t].copy()
                imag = frames[b, t].copy() if t < H else ar[b, t - H].copy()
                for img, tag in ((real, "REAL"), (imag, "NanoWM" if t >= H else "context")):
                    cv2.putText(
                        img, f"{tag} b={be:.2f} s{s}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                    )
                    cv2.putText(
                        img, f"t={t * FI / 10:.1f}s", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                    )
                line.append(np.hstack([real, imag, np.full((IMG, 6, 3), 90, np.uint8)]))
            grid.append(np.hstack(line))
        video.append(np.vstack(grid))
    video = [v[: v.shape[0] // 2 * 2, : v.shape[1] // 2 * 2] for v in video]
    imageio.mimwrite(work / "real_vs_nanowm_grid.mp4", video, fps=2, codec="libx264")
    print(
        f"wrote {work / 'real_vs_nanowm_grid.mp4'} (rows = beta {betas.tolist()}, cols = starts {good_starts}, each cell REAL | NanoWM)"
    )


def stage_video(args) -> None:
    """One video per start: columns = beta, top row REAL, bottom row NanoWM (autoregressive), with
    each column's block error for that round. Plus a keyframe PNG (rows = beta x {REAL, NanoWM},
    columns = time). Same start + same random-target sequence in every column, so the columns
    differ only by beta."""
    import cv2
    import imageio.v2 as imageio

    work = args.work
    d = np.load(work / "real.npz")
    frames, beta, sid, pen, states = d["frames"], d["beta"], d["sid"], d["wall_pen"], d["states"]
    ar = np.load(work / "wm_ar.npz")["imag"]
    fm_path = work / "fidelity_measures.npz"
    fm = np.load(fm_path) if fm_path.exists() else None
    R = ar.shape[1]
    unit = json.loads(str(d["meta"])).get("unit", "beta")
    betas = np.unique(beta)
    starts = (
        args.video_starts
        or [int(s) for s in np.unique(sid) if (pen[sid == s] <= 10).all()][: args.n_video_starts]
    )
    out = work / "beta_videos"
    out.mkdir(exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def bar(width: int, text: str, h: int = 26, shade: int = 235) -> np.ndarray:
        b = np.full((h, width, 3), shade, np.uint8)
        cv2.putText(b, text, (6, h - 8), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return b

    def tag(img: np.ndarray, text: str, color=(0, 0, 0)) -> np.ndarray:
        img = img.copy()
        cv2.putText(img, text, (5, 16), font, 0.45, color, 1, cv2.LINE_AA)
        return img

    vsep = np.full((2 * IMG + 26 * 3, 4, 3), 60, np.uint8)
    for s in starts:
        idx = [int(np.nonzero((sid == s) & (beta == be))[0][0]) for be in betas]
        video = []
        for t in range(H + R):
            cols = []
            for b, be in zip(idx, betas, strict=True):
                real = tag(frames[b, t], "REAL")
                if t < H:
                    imag, foot = (
                        tag(frames[b, t], "context (real)", (90, 90, 90)),
                        "NanoWM gets this real frame",
                    )
                else:
                    imag, foot = tag(ar[b, t - H], "NanoWM", (180, 0, 0)), ""
                    if fm is not None:
                        rm, im = fm["real"][b, t - H], fm["imag"][b, t - H]
                        ang = np.degrees(abs(np.angle(np.exp(1j * (im[2] - rm[2])))))
                        foot = f"block err {np.hypot(*(im[:2] - rm[:2])):.0f}px  angle {ang:.0f}deg"
                head = f"noise = {be:.2f} sigma" if unit.startswith("k") else f"beta = {be:.2f}"
                cols.append(np.vstack([bar(IMG, head, shade=210), real, bar(IMG, ""), imag, bar(IMG, foot)]))
                cols.append(vsep)
            grid = np.hstack(cols[:-1])
            phase = (
                "context: real frames fed to NanoWM"
                if t < H
                else f"NanoWM round {t - H} (open loop, same actions as REAL)"
            )
            frame = np.vstack(
                [bar(grid.shape[1], f"start s{s}   t = {t * FI / 10:.1f} s   {phase}", h=30, shade=255), grid]
            )
            video.append(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2])
        video += [video[-1]] * 3  # hold the last frame
        mp4 = out / f"beta_real_vs_nanowm_s{s:02d}.mp4"
        imageio.mimwrite(mp4, video, fps=2, codec="libx264")

        # keyframes: rows = beta x {REAL, NanoWM}, cols = time
        ts = [H - 1] + [H + r for r in (3, 7, 11, R - 1) if r < R]
        rows = [
            bar(
                120 + len(ts) * (IMG + 4),
                "            " + "".join(f"t = {t * FI / 10:.1f} s".ljust(33) for t in ts),
                h=30,
                shade=255,
            )
        ]
        for b, be in zip(idx, betas, strict=True):
            for kind in ("REAL", "NanoWM"):
                label = np.full((IMG, 120, 3), 235, np.uint8)
                cv2.putText(
                    label,
                    (f"{be:.2f} sigma" if unit.startswith("k") else f"beta={be:.2f}"),
                    (6, IMG // 2 - 6),
                    font,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    label,
                    kind,
                    (6, IMG // 2 + 16),
                    font,
                    0.5,
                    (180, 0, 0) if kind == "NanoWM" else (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cells = [label]
                for t in ts:
                    img = frames[b, t] if (kind == "REAL" or t < H) else ar[b, t - H]
                    cells += [img, np.full((IMG, 4, 3), 60, np.uint8)]
                rows.append(np.hstack(cells))
            rows.append(np.full((6, rows[-1].shape[1], 3), 60, np.uint8))
        cv2.imwrite(
            str(out / f"beta_keyframes_s{s:02d}.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR)
        )

        yf = [states[i, -1, 3] for i in idx]
        errs = [
            np.hypot(*(fm["imag"][i, -1, :2] - fm["real"][i, -1, :2])) if fm is not None else np.nan
            for i in idx
        ]
        print(
            f"s{s:02d}: start CoG y {states[idx[0], 0, 3]:.0f} | REAL final CoG y by beta "
            + ", ".join(f"{be:.2f}:{y:.0f}" for be, y in zip(betas, yf, strict=True))
            + " | NanoWM final block err "
            + ", ".join(f"{e:.0f}px" for e in errs)
            + f"  -> {mp4.name}"
        )


def stage_analyze(args) -> None:
    works = [args.work] + ([args.ref] if args.ref else [])
    res = {}
    for w in works:
        if not (w / "wm_ar.npz").exists():
            print(f"skip {w}: no wm_ar.npz")
            continue
        r = analyze_one(w)
        res[r["region"]] = r
        (w / "fidelity_summary.json").write_text(json.dumps(r, indent=1))
        print(f"\n=== {r['region']}  branches={r['branches']}  wall-pen excluded={r['wall_pen_excluded']}  "
              f"estimator CoG err on real frames {r['estimator_cog_err_px']:.1f}px ===")  # fmt: skip
        R = r["rounds"]
        show = sorted({x for x in (0, 3, 7, 11, R - 1) if x < R})
        print(f"{r['unit']:>9} {'metric':>10} " + " ".join(f"{'r' + str(x) + ' WM/copy':>14}" for x in show))
        for be, row in r["by_beta"].items():
            pr = row["per_round"]
            for k in ("block_px", "angle_deg", "agent_px", "cov_abs"):
                cells = " ".join(f"{pr[k][x]:6.1f}/{pr['copy_' + k][x]:<7.1f}" for x in show)
                print(f"{be:>5} {k:>10} {cells}")
            print(f"{be:>5} {'vanished':>10} " + " ".join(f"{pr['vanished'][x]:14.2f}" for x in show))
            nm = row["net_motion"]
            print(f"{'':>5} net motion @r{R - 1}: real moved>20px {nm['real_moved_gt20px']:.2f}, disp err {nm['disp_err_px_median']:.0f}px, "
                  f"corr(dy) {nm['corr_dy'] if nm['corr_dy'] is None else round(nm['corr_dy'], 2)}, WM beats copy {nm['wm_beats_copy_final_block']:.2f}")  # fmt: skip
            if "teacher_forced" in row:
                tf = row["teacher_forced"]
                print(
                    f"{'':>5} teacher-forced 1-step block_px @rounds {tf['rounds']}: {tf['block_px']}  angle {tf['angle_deg']}"
                )
        if args.video:
            write_video(w)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing -> no figure")
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for (region, r), ls in zip(res.items(), ["-", ":"], strict=False):
        for i, (be, row) in enumerate(r["by_beta"].items()):
            for ax, k in zip(axes, ("block_px", "angle_deg", "agent_px", "cov_abs"), strict=True):
                ax.plot(row["per_round"][k], ls, color=f"C{i}", label=f"{region} b={be}")
                if ls == "-":
                    ax.plot(row["per_round"]["copy_" + k], "--", color=f"C{i}", alpha=0.4)
    for ax, k in zip(
        axes,
        ("block CoG err (px/512)", "block angle err (deg)", "agent err (px)", "|coverage err|"),
        strict=True,
    ):
        ax.set_title(k)
        ax.set_xlabel("NanoWM round (0.5 s each)")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "NanoWM vs real: solid = region, dotted = reference region, dashed = copy-last-context baseline"
    )
    fig.tight_layout()
    png = args.work / "fidelity_vs_round.png"
    fig.savefig(png, dpi=110)
    print(f"wrote {png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["real", "wm", "analyze", "video"])
    p.add_argument("--work", type=Path, default=Path("examples/nanowm_pusht_aug/cl/wmfid/topband"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # real
    p.add_argument("--region", choices=["topband", "reset"], default="topband")
    p.add_argument("--ckpt", type=Path, default=BASE_CKPT)
    p.add_argument("--n-starts", type=int, default=48)
    p.add_argument("--inject", choices=["sigma", "beta"], default="sigma", help="noise parameterization")
    p.add_argument("--noise-sigma", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                   help="--inject sigma: noise per step in units of NanoWM's training-action sigma (~20.8 px)")  # fmt: skip
    p.add_argument("--noise-hold", type=int, default=FI, help="--inject sigma: steps each noise draw is held")
    p.add_argument("--max-z", type=float, default=3.0, help="--inject sigma: cap on |action| in sigma units")
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0], help="--inject beta only")
    p.add_argument("--rounds", type=int, default=16, help="NanoWM rounds (0.5 s each)")
    p.add_argument("--rand-hold", type=int, default=8, help="--inject beta: steps each random target is held")
    p.add_argument("--seed", type=int, default=300_000)
    # wm
    p.add_argument("--mode", choices=["ar", "tf"], default="ar")
    p.add_argument("--tf-rounds", type=int, nargs="+", default=[0, 5, 10, 15])
    p.add_argument("--num-sampling-steps", type=int, default=30)
    p.add_argument("--wm-batch", type=int, default=24)
    # analyze
    p.add_argument("--ref", type=Path, default=None, help="work dir of the in-distribution reference run")
    p.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    # video
    p.add_argument(
        "--video-starts", type=int, nargs="*", default=None, help="start ids (default: first clean ones)"
    )
    p.add_argument("--n-video-starts", type=int, default=6)
    args = p.parse_args()
    {"real": stage_real, "wm": stage_wm, "analyze": stage_analyze, "video": stage_video}[args.stage](args)


if __name__ == "__main__":
    main()
