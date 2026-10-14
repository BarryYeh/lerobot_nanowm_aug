#!/usr/bin/env python
"""Does NanoWM, given the SAME context and the SAME actions, produce different AND sharp futures per seed?

Every SAILOR candidate so far differed in its actions, so the world model's own test-time diversity
-- the property that sets a diffusion WM apart from a deterministic MSE model (DINO-WM, LeWM) -- was
never measured in isolation. Here the only thing that varies between samples is the WM's noise.

Setup (lerobot venv driver, nanowm venv worker -- the two-venv split of README.md):

  real    per anchor (a body-origin seed state, default the SAILOR v1 test anchors) the base policy
          runs in gym_pusht through `closed_loop_rollout.run_lead_in` -- the SAME lead-in as the
          closed loop, just extended past the anchor: the first (H-1)*fi steps give the H real context
          frames, the remaining (R-1)*fi + nf*fi - (H-1)*fi steps are the action sequence that is then
          REPLAYED (not re-sampled) to NanoWM, and the sim frames every fi steps are the ground truth.
          Windows follow the closed_loop_rollout ALIGNMENT CONTRACT (round r = stream[r*fi : r*fi+nf*fi],
          generating the frame at stream step (H+r)*fi) with the exact relative action (target minus the
          SIMULATED agent position before it, as in wm_fidelity_ood) instead of the pd_step estimate.
  wm      each anchor is replicated S times in one worker batch (`imagined_rollout.py closed-loop
          --seed`): replicas get independent noise rows of one seeded RNG stream, i.e. S WM seeds.
          Arms: `main` (num_sampling_steps from configs/nanowm_pusht_common.yaml = 30, all anchors) and
          `steps250` (250 steps, first --n-anchors-250 anchors) as a "do 30 steps blur it" control.
  recon   SD-VAE encode -> decode of every sim frame through the worker's own paths
          (`imagined_rollout.py recon`): what a PERFECT world model would show, separating the VAE's
          blur from the generator's.
  detcheck  tiny reruns: same --wm-seed twice must be bit-identical, a different seed must differ.
  analyze per round: (a) spread across seeds of block CoG / angle / coverage (frame estimator =
          wm_fidelity_ood.measure, also applied to the sim frames), (b) sharpness of generated vs
          VAE-recon vs sim frames, (c) whether the sim truth falls inside the S samples' range and the
          error of the sample mean; plus grids (S seeds side by side + sim + recon).
          -> <out>/summary.json, <out>/summary.md, <out>/grids/*.png

    P=examples/nanowm_pusht_aug
    CUDA_VISIBLE_DEVICES=7 uv run --extra training python $P/eval/wm_seed_diversity.py all \
        --out ~/datastorage/nanowm_pusht_aug/wm_seed_diversity
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

_EVAL = Path(__file__).resolve().parent
_PROJ = _EVAL.parent  # examples/nanowm_pusht_aug
_REPO = _PROJ.parents[1]
for _p in (_PROJ, _EVAL, _PROJ / "attribution"):
    sys.path.insert(0, str(_p))
from closed_loop_rollout import NANOWM_PY, run_lead_in  # noqa: E402
from config_io import load_config  # noqa: E402
from datagen import action_convert as ac  # noqa: E402
from datagen.coverage_from_frame import block_mask  # noqa: E402
from nwaug.pusht_state import load_seeds  # noqa: E402
from wm_fidelity_ood import errors, measure, run_worker  # noqa: E402

COMMON = load_config(_REPO / "configs" / "nanowm_pusht_common.yaml")
H = int(COMMON["nanowm"]["history"])  # 3
NF = int(COMMON["nanowm"]["num_frames"])  # 4
FI = int(COMMON["pusht"]["frame_interval"])  # 5
DEFAULT_ROUNDS = int(COMMON["nanowm"]["rounds"])  # 8
DEFAULT_STEPS = int(COMMON["nanowm"]["num_sampling_steps"])  # 30
DEFAULT_SEEDS = Path("~/datastorage/nanowm_pusht_aug/value_model/sailor_v1/seeds_test.npz").expanduser()
# A sample counts as "correct" when its block is within these of the sim truth (and not vanished).
OK_BLOCK_PX, OK_ANGLE_DEG = 20.0, 15.0  # sim px (T stem is 30 px wide); estimator floor is a few px
# "blurry" = block edge width > this multiple of the VAE recon of the same sim frame. Edge width is
# contrast-normalized; the band-Laplacian ratio is reported too but is content-dependent (goal overlap,
# outline colour) and flagged visibly sharp frames, so it is not used to classify.
BLUR_EW_RATIO = 1.3
MOVED_PX = 20.0  # an anchor is a "contact" anchor if the sim block moved more than this over the horizon


def stream_len(rounds: int) -> int:
    """Stream steps (from the FIRST context frame) the last round's window needs: (R-1)*fi + nf*fi."""
    return (rounds - 1) * FI + NF * FI


def replay_windows(acts: np.ndarray, apos: np.ndarray, rounds: int) -> np.ndarray:
    """acts / apos (T,2) absolute targets and the simulated agent position BEFORE each -> (R,nf,2fi).

    Alignment contract of closed_loop_rollout: round r's window is stream[r*fi : r*fi + nf*fi], the
    stream indexed from the first context frame. rel = target - agent (exact, from the simulator)."""
    need = NF * FI
    assert len(acts) >= stream_len(rounds) and len(apos) >= stream_len(rounds), (len(acts), rounds)
    return np.stack(
        [ac.rel_to_nanowm(acts[r * FI : r * FI + need] - apos[r * FI : r * FI + need]) for r in range(rounds)]
    ).astype(np.float32)


# ------------------------------------------------------------------------------------------- real
def stage_real(args) -> None:
    import gym_pusht  # noqa: F401
    import gymnasium as gym
    import torch

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dev, R = args.device, args.rounds
    T = stream_len(R)
    states = load_seeds(args.seeds, "body")[: args.n_anchors]
    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )

    def policy_act(i: int, t: int, pix: np.ndarray) -> np.ndarray:  # as closed_loop_rollout.policy_act
        if t == 0:
            policy.reset()
        img = torch.from_numpy(pix).permute(2, 0, 1).float().div(255.0)[None].to(dev)
        with torch.no_grad():
            a = post(policy.select_action(pre({"observation.image": img})))
        return a[0].cpu().numpy().astype(np.float64)

    def mk(obs_type: str, size: int):
        return gym.make(
            "gym_pusht/PushT-v0", obs_type=obs_type, observation_width=size, observation_height=size,
            max_episode_steps=T + 40,
        )  # fmt: skip

    env96, env224 = mk("pixels_agent_pos", 96), mk("pixels", 224)
    torch.manual_seed(args.torch_seed)
    # `lead` = the whole replayed stream; frames every FI steps: H context + R ground-truth frames
    res = run_lead_in(env96, env224, states, policy_act, hist=H + R, fi=FI, lead=T)
    env96.close()
    env224.close()
    frames = res["ctx"]  # (n, H+R, 224,224,3): [:H] context, [H+r] = sim truth of round r
    windows = np.stack(
        [replay_windows(res["lead_acts"][i], res["lead_pos"][i], R) for i in range(len(states))]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "real.npz",
        frames=frames, acts=res["lead_acts"], apos=res["lead_pos"], avel=res["lead_vel"], windows=windows,
        seed_states=states, meta=json.dumps({"ckpt": str(args.ckpt), "seeds": str(args.seeds), "rounds": R,
                                             "H": H, "fi": FI, "nf": NF, "torch_seed": args.torch_seed}),
    )  # fmt: skip
    m0 = np.array([measure(frames[i, H - 1]) for i in range(len(states))])
    m1 = np.array([measure(frames[i, H + R - 1]) for i in range(len(states))])
    disp = np.hypot(*(m1[:, :2] - m0[:, :2]).T)
    print(
        f"real: {len(states)} anchors, stream {T} steps; block moved over the horizon: median {np.nanmedian(disp):.0f} sim px, >20px in {np.mean(disp > 20):.0%}"
    )
    print(f"wrote {args.out / 'real.npz'}")


# --------------------------------------------------------------------------------------------- wm
def _arms(args) -> list[tuple[str, int, int]]:
    """(name, num_sampling_steps, n_anchors)."""
    arms = [("main", args.num_sampling_steps, args.n_anchors)]
    if args.n_anchors_250 > 0:
        arms.append(("steps250", 250, min(args.n_anchors_250, args.n_anchors)))
    return arms


def stage_wm(args) -> None:
    d = np.load(args.out / "real.npz")
    frames, windows = d["frames"], d["windows"]
    S = args.n_wm_seeds
    for name, steps, na in _arms(args):
        if args.arm and name not in args.arm:
            continue
        seeds = np.repeat(frames[:na, :H], S, axis=0)  # anchor-major: row a*S + s
        win = np.repeat(windows[:na], S, axis=0)
        wargs = SimpleNamespace(num_sampling_steps=steps, wm_batch=args.wm_batch, wm_seed=args.wm_seed)
        print(f"wm[{name}]: {na} anchors x {S} seeds = {na * S} branches, {steps} steps, seed {args.wm_seed}")
        imag = run_worker(args.out / f"worker_{name}", seeds, win, wargs)  # (na*S, R, 224,224,3)
        imag = imag.reshape(na, S, *imag.shape[1:])
        np.savez_compressed(
            args.out / f"wm_{name}.npz", imag=imag, steps=np.int64(steps), wm_seed=np.int64(args.wm_seed)
        )
        print(f"wrote {args.out / f'wm_{name}.npz'}")


def stage_recon(args) -> None:
    d = np.load(args.out / "real.npz")
    np.savez(args.out / "recon_in.npz", frames_224=d["frames"])
    cmd = [str(NANOWM_PY), str(_PROJ / "imagined_rollout.py"), "recon", "--in", str(args.out / "recon_in.npz"),
           "--out", str(args.out / "recon.npz"), "--seed", str(args.wm_seed)]  # fmt: skip
    subprocess.run(cmd, check=True)
    (args.out / "recon_in.npz").unlink()


def stage_detcheck(args) -> None:
    """Same seed twice -> bit-identical; other seed -> different; replicas within a run -> different."""
    d = np.load(args.out / "real.npz")
    na, S, R = 2, 2, 2
    seeds = np.repeat(d["frames"][:na, :H], S, axis=0)
    win = np.repeat(d["windows"][:na, :R], S, axis=0)
    out = {}
    runs = {}
    for tag, sd in (("a", args.wm_seed), ("b", args.wm_seed), ("c", args.wm_seed + 1)):
        wargs = SimpleNamespace(
            num_sampling_steps=args.num_sampling_steps, wm_batch=args.wm_batch, wm_seed=sd
        )
        runs[tag] = run_worker(args.out / f"worker_det_{tag}", seeds, win, wargs).astype(np.int16)
    out["same_seed_max_abs_diff"] = int(np.abs(runs["a"] - runs["b"]).max())
    out["other_seed_mean_abs_diff"] = float(np.abs(runs["a"] - runs["c"]).mean())
    out["replicas_mean_abs_diff"] = float(np.abs(runs["a"][0::S] - runs["a"][1::S]).mean())
    (args.out / "detcheck.json").write_text(json.dumps(out, indent=2))
    print("detcheck:", out)


# ---------------------------------------------------------------------------------------- analyze
def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def sharpness(rgb: np.ndarray) -> np.ndarray:
    """224px frame -> [laplacian variance (whole frame), laplacian variance in a band around the block
    edge, block edge width proxy (px) = edge contrast / p90 gradient magnitude in that band].

    The band / contrast come from the frame's OWN block mask, so a misplaced block is still scored on
    its edges. Edge width ~1 px for a crisp raster edge, grows linearly with blur; nan if no block."""
    g = _gray(rgb)
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
    m = block_mask(rgb)
    if m.sum() < 50:
        return np.array([lap.var(), np.nan, np.nan])
    k = np.ones((3, 3), np.uint8)
    dil, ero = cv2.dilate(m, k, iterations=3), cv2.erode(m, k, iterations=3)
    band = (dil > 0) & (ero == 0)
    ring = (cv2.dilate(m, k, iterations=6) > 0) & (dil == 0)
    inner = ero > 0
    gx, gy = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3) / 8.0, cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    grad = np.hypot(gx, gy)
    contrast = abs(np.median(g[inner]) - np.median(g[ring])) if inner.any() and ring.any() else np.nan
    p90 = np.percentile(grad[band], 90)
    return np.array([lap.var(), lap[band].var(), contrast / max(p90, 1e-6)])


def circ_std_deg(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return np.nan
    rbar = np.abs(np.exp(1j * a).mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(rbar, 1e-12)))))


def circ_mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.angle(np.exp(1j * a).mean())) if len(a) else np.nan


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


def in_range(samples: np.ndarray, truth: float) -> float:
    s = samples[np.isfinite(samples)]
    if len(s) < 2 or not np.isfinite(truth):
        return np.nan
    return float(s.min() <= truth <= s.max())


def boot_ci(per_anchor: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over anchors with a percentile bootstrap 95% CI (resampling anchors)."""
    x = np.asarray(per_anchor, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return (float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def per_sample(imag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(na,S,R,224,224,3) -> measures (na,S,R,7), sharpness (na,S,R,3)."""
    na, S, R = imag.shape[:3]
    me = np.array([[[measure(imag[a, s, r]) for r in range(R)] for s in range(S)] for a in range(na)])
    sh = np.array([[[sharpness(imag[a, s, r]) for r in range(R)] for s in range(S)] for a in range(na)])
    return me, sh


def _pos_sd(me: np.ndarray) -> np.ndarray:
    """(...,S,7) -> (...) spread of the block CoG over seeds, sqrt(var x + var y) in sim px."""
    return np.sqrt(np.nanvar(me[..., 0], -1) + np.nanvar(me[..., 1], -1))


def _truth_in_range(me_r: np.ndarray, truth_r: np.ndarray) -> dict[str, np.ndarray]:
    """me_r (na,S,7) samples, truth_r (na,7) -> per-anchor 0/1 for x, y, angle (about the samples'
    circular mean), coverage. With S exchangeable samples a calibrated model scores (S-1)/(S+1) per dim."""
    na = len(me_r)
    mu = np.array([circ_mean(me_r[a, :, 2]) for a in range(na)])
    return {
        "x": np.array([in_range(me_r[a, :, 0], truth_r[a, 0]) for a in range(na)]),
        "y": np.array([in_range(me_r[a, :, 1], truth_r[a, 1]) for a in range(na)]),
        "angle": np.array(
            [in_range(wrap(me_r[a, :, 2] - mu[a]), wrap(truth_r[a, 2] - mu[a])) for a in range(na)]
        ),
        "cov": np.array([in_range(me_r[a, :, 5], truth_r[a, 5]) for a in range(na)]),
    }


def arm_metrics(imag, me, sh, real_m, recon_m, recon_sh, sim_sh, moved) -> dict:
    """Per-round tables for one arm. real_m / recon_m: (na,R,7) measures of sim / VAE-recon frames;
    moved (na,) bool = the block moved > MOVED_PX in the sim over the horizon (contact anchors)."""
    na, S, R = me.shape[:3]
    real_b = np.repeat(real_m[:, None], S, 1)  # (na,S,R,7)
    err = errors(me, real_b)  # each (na,S,R)
    floor = errors(recon_m, real_m)  # VAE + estimator floor, (na,R)
    ew_ratio = sh[..., 2] / recon_sh[:, None, :, 2]  # (na,S,R) edge width gen / recon of the same sim frame
    rows = []
    for r in range(R):
        pos_sd = _pos_sd(me[:, :, r])
        ang_sd = np.array([circ_std_deg(me[a, :, r, 2]) for a in range(na)])
        cov_sd = np.nanstd(me[:, :, r, 5], 1)
        agent_sd = np.sqrt(np.nanvar(me[:, :, r, 3], 1) + np.nanvar(me[:, :, r, 4], 1))
        g = np.stack([[_gray(imag[a, s, r]) for s in range(S)] for a in range(na)])  # (na,S,224,224)
        pix_sd = g.std(1).mean((1, 2))
        # sample mean vs truth, and is the truth inside the samples' range?
        mean_err = np.hypot(*(np.nanmean(me[:, :, r, :2], 1) - real_m[:, r, :2]).T)
        mean_ang = np.array([circ_mean(me[a, :, r, 2]) for a in range(na)])
        mean_ang_err = np.degrees(np.abs(wrap(mean_ang - real_m[:, r, 2])))
        mean_cov_err = np.abs(np.nanmean(me[:, :, r, 5], 1) - real_m[:, r, 5])
        # systematic miss: the seeds agree with each other but not with the sim
        systematic = (mean_err > 2 * pos_sd) & (mean_err > OK_BLOCK_PX)
        tin_sim = _truth_in_range(me[:, :, r], real_m[:, r])
        tin_rec = _truth_in_range(me[:, :, r], recon_m[:, r])
        ok = (
            (err["block_px"][:, :, r] < OK_BLOCK_PX)
            & (err["angle_deg"][:, :, r] < OK_ANGLE_DEG)
            & (err["vanished"][:, :, r] == 0)
        )
        vanished = err["vanished"][:, :, r] == 1
        blurry = (ew_ratio[:, :, r] > BLUR_EW_RATIO) & ~vanished  # nan ratio (no block) -> not blurry
        sharp = ~blurry & ~vanished
        e_s = err["block_px"][:, :, r]
        rows.append({
            "round": r, "t_s": (r + 1) * FI / 10,
            "spread": {"block_pos_sd_px": boot_ci(pos_sd), "block_ang_sd_deg": boot_ci(ang_sd),
                       "cov_sd": boot_ci(cov_sd), "agent_pos_sd_px": boot_ci(agent_sd), "pixel_sd_gray": boot_ci(pix_sd),
                       "block_pos_sd_px_median_anchor": float(np.nanmedian(pos_sd)),
                       "block_pos_sd_px_moved": boot_ci(pos_sd[moved]), "block_pos_sd_px_static": boot_ci(pos_sd[~moved]),
                       "block_ang_sd_deg_moved": boot_ci(ang_sd[moved]), "block_ang_sd_deg_static": boot_ci(ang_sd[~moved]),
                       "agent_pos_sd_px_moved": boot_ci(agent_sd[moved])},
            "error": {"block_px_sample": boot_ci(np.nanmean(e_s, 1)),
                      "block_px_sample_moved": boot_ci(np.nanmean(e_s[moved], 1)),
                      "block_px_sample_static": boot_ci(np.nanmean(e_s[~moved], 1)),
                      "angle_deg_sample": boot_ci(np.nanmean(err["angle_deg"][:, :, r], 1)),
                      "cov_abs_sample": boot_ci(np.nanmean(err["cov_abs"][:, :, r], 1)),
                      "vanished_rate": boot_ci(err["vanished"][:, :, r].mean(1)),
                      "block_px_of_mean": boot_ci(mean_err), "angle_deg_of_mean": boot_ci(mean_ang_err),
                      "cov_abs_of_mean": boot_ci(mean_cov_err), "systematic_miss_anchor_frac": boot_ci(systematic.astype(float))},
            "floor_vae_recon": {"block_px": boot_ci(floor["block_px"][:, r]), "angle_deg": boot_ci(floor["angle_deg"][:, r]),
                                "cov_abs": boot_ci(floor["cov_abs"][:, r])},
            "truth_in_range": {k: boot_ci(v) for k, v in tin_sim.items()},
            "recon_truth_in_range": {k: boot_ci(v) for k, v in tin_rec.items()},
            "sharp": {"lapvar_gen": boot_ci(np.nanmean(sh[:, :, r, 0], 1)), "lapvar_recon": boot_ci(recon_sh[:, r, 0]),
                      "lapvar_sim": boot_ci(sim_sh[:, r, 0]),
                      "band_lapvar_ratio_gen_over_recon": boot_ci(np.nanmean(sh[:, :, r, 1] / recon_sh[:, None, r, 1], 1)),
                      "edge_width_gen_px": boot_ci(np.nanmean(sh[:, :, r, 2], 1)), "edge_width_recon_px": boot_ci(recon_sh[:, r, 2]),
                      "edge_width_sim_px": boot_ci(sim_sh[:, r, 2]),
                      "edge_width_ratio_gen_over_recon": boot_ci(np.nanmean(ew_ratio[:, :, r], 1)),
                      "edge_width_ratio_p99": float(np.nanpercentile(ew_ratio[:, :, r], 99))},
            "types": {"sharp_correct": boot_ci((sharp & ok).mean(1)), "sharp_wrong": boot_ci((sharp & ~ok).mean(1)),
                      "blurry": boot_ci(blurry.mean(1)), "vanished": boot_ci(vanished.mean(1))},
        })  # fmt: skip
    last = R - 1
    per_anchor = [
        {"anchor": a, "moved": bool(moved[a]), "block_pos_sd_px": float(_pos_sd(me[a, :, last])),
         "block_ang_sd_deg": circ_std_deg(me[a, :, last, 2]),
         "err_sample_px_mean": float(np.nanmean(err["block_px"][a, :, last])),
         "err_sample_px_max": float(np.nanmax(err["block_px"][a, :, last])),
         "err_of_mean_px": float(np.hypot(*(np.nanmean(me[a, :, last, :2], 0) - real_m[a, last, :2]))),
         "frac_wrong": float(np.mean(~((err["block_px"][a, :, last] < OK_BLOCK_PX) & (err["angle_deg"][a, :, last] < OK_ANGLE_DEG))))}
        for a in range(na)
    ]  # fmt: skip
    return {
        "n_anchors": int(na), "n_seeds": int(S), "n_moved": int(moved.sum()), "per_round": rows,
        "per_anchor_last_round": per_anchor,
        "edge_width_ratio_pooled_pctl": dict(zip(["p50", "p90", "p99", "max"], np.nanpercentile(ew_ratio, [50, 90, 99, 100]).round(3).tolist(), strict=True)),
    }  # fmt: skip


def make_grid(imag_a, sim_a, recon_a, rounds, label: str) -> np.ndarray:
    """One anchor: rows = rounds, columns = sim | recon | seed 0..S-1."""
    S = imag_a.shape[0]
    rows = []
    for r in rounds:
        cells = [sim_a[H + r].copy(), recon_a[H + r].copy()] + [imag_a[s, r].copy() for s in range(S)]
        tags = ["SIM", "VAE recon"] + [f"seed {s}" for s in range(S)]
        for c, tg in zip(cells, tags, strict=True):
            cv2.putText(
                c, f"{tg}  +{(r + 1) * FI / 10:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )
        rows.append(
            np.concatenate([np.pad(c, ((2, 2), (2, 2), (0, 0)), constant_values=255) for c in cells], 1)
        )
    g = np.concatenate(rows, 0)
    bar = np.full((22, g.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return np.concatenate([bar, g], 0)


def make_zoom(
    imag_a, sim_a, recon_a, r: int, cx: float, cy: float, n_seeds: int = 6, half: int = 40, scale: int = 3
):
    """Block-centred crops (x`scale`, nearest) of sim | recon | first seeds, to eyeball edge blur."""
    x0 = int(np.clip(round(cx) - half, 0, 224 - 2 * half))
    y0 = int(np.clip(round(cy) - half, 0, 224 - 2 * half))
    imgs = [sim_a[H + r], recon_a[H + r]] + [imag_a[s, r] for s in range(min(n_seeds, len(imag_a)))]
    crops = [
        cv2.resize(
            i[y0 : y0 + 2 * half, x0 : x0 + 2 * half],
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        for i in imgs
    ]
    return np.concatenate([np.pad(c, ((2, 2), (2, 2), (0, 0)), constant_values=255) for c in crops], 1)


def _fmt(ci, nd=1) -> str:
    m, lo, hi = ci
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if np.isfinite(m) else "n/a"


def summary_md(res: dict) -> str:
    out = [
        f"anchors moved > {MOVED_PX:.0f} sim px over the horizon: see n_moved per arm. "
        f"correct = block < {OK_BLOCK_PX:.0f} px and < {OK_ANGLE_DEG:.0f} deg from the sim; "
        f"blurry = block edge width > {BLUR_EW_RATIO} x the VAE recon of the same sim frame. "
        "Values: mean over anchors [bootstrap 95% CI over anchors]. Distances in sim px (512 px workspace).\n"
    ]
    for arm, a in res["arms"].items():
        out.append(
            f"## arm `{arm}`  ({a['n_anchors']} anchors ({a['n_moved']} moved) x {a['n_seeds']} seeds, {a['steps']} steps)\n"
        )
        out.append("### spread across seeds vs error to the sim\n")
        out.append(
            "| round (+s) | block pos sd | (moved / static) | angle sd deg | cov sd | agent sd | err/sample | err of seed mean | VAE floor | systematic-miss anchors |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in a["per_round"]:
            sp, er, fl = row["spread"], row["error"], row["floor_vae_recon"]
            out.append(
                f"| {row['round']} (+{row['t_s']:.1f}) | {_fmt(sp['block_pos_sd_px'])} | {sp['block_pos_sd_px_moved'][0]:.1f} / {sp['block_pos_sd_px_static'][0]:.1f} | "
                f"{_fmt(sp['block_ang_sd_deg'])} | {_fmt(sp['cov_sd'], 3)} | {_fmt(sp['agent_pos_sd_px'])} | {_fmt(er['block_px_sample'])} | "
                f"{_fmt(er['block_px_of_mean'])} | {_fmt(fl['block_px'])} | {_fmt(er['systematic_miss_anchor_frac'], 2)} |"
            )  # fmt: skip
        out.append("\n### truth inside the seeds' range (calibrated: (S-1)/(S+1) per dim)\n")
        out.append("| round | x (sim truth) | y | angle | cov | x (recon truth) | y | angle |")
        out.append("|---|---|---|---|---|---|---|---|")
        for row in a["per_round"]:
            t, tr = row["truth_in_range"], row["recon_truth_in_range"]
            out.append(f"| {row['round']} | {_fmt(t['x'], 2)} | {_fmt(t['y'], 2)} | {_fmt(t['angle'], 2)} | {_fmt(t['cov'], 2)} | "
                       f"{_fmt(tr['x'], 2)} | {_fmt(tr['y'], 2)} | {_fmt(tr['angle'], 2)} |")  # fmt: skip
        out.append("\n### sharpness and error type\n")
        out.append(
            "| round | edge width gen px | recon px | sim px | gen/recon | lapvar gen / recon / sim | sharp-correct | sharp-wrong | blurry | vanished |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in a["per_round"]:
            sh, ty = row["sharp"], row["types"]
            out.append(
                f"| {row['round']} | {_fmt(sh['edge_width_gen_px'], 2)} | {_fmt(sh['edge_width_recon_px'], 2)} | {_fmt(sh['edge_width_sim_px'], 2)} | "
                f"{_fmt(sh['edge_width_ratio_gen_over_recon'], 3)} | {sh['lapvar_gen'][0]:.0f} / {sh['lapvar_recon'][0]:.0f} / {sh['lapvar_sim'][0]:.0f} | "
                f"{_fmt(ty['sharp_correct'], 2)} | {_fmt(ty['sharp_wrong'], 2)} | {_fmt(ty['blurry'], 3)} | {_fmt(ty['vanished'], 3)} |"
            )  # fmt: skip
        out.append(
            f"\nedge width gen/recon, pooled per-sample percentiles: {a['edge_width_ratio_pooled_pctl']}\n"
        )
        out.append("### per anchor, last round\n")
        out.append("| anchor | moved | pos sd | ang sd | err/sample mean | max | err of mean | frac wrong |")
        out.append("|---|---|---|---|---|---|---|---|")
        for q in a["per_anchor_last_round"]:
            out.append(f"| {q['anchor']} | {'y' if q['moved'] else 'n'} | {q['block_pos_sd_px']:.1f} | {q['block_ang_sd_deg']:.1f} | "
                       f"{q['err_sample_px_mean']:.1f} | {q['err_sample_px_max']:.1f} | {q['err_of_mean_px']:.1f} | {q['frac_wrong']:.2f} |")  # fmt: skip
        out.append("")
    if "paired_250_vs_30" in res:
        out.append("## 250 vs 30 steps (same anchors, paired; 250 minus 30)\n")
        for k, v in res["paired_250_vs_30"].items():
            out.append(f"- {k}: {_fmt(v, 3)}")
    if "detcheck" in res:
        out.append(f"\n## determinism check\n\n{res['detcheck']}")
    return "\n".join(out) + "\n"


def stage_analyze(args) -> None:
    d = np.load(args.out / "real.npz")
    sim = d["frames"]
    recon = np.load(args.out / "recon.npz")["frames"]
    n, Ht = sim.shape[:2]
    R = Ht - H
    real_m = np.array([[measure(sim[a, H + r]) for r in range(R)] for a in range(n)])
    recon_m = np.array([[measure(recon[a, H + r]) for r in range(R)] for a in range(n)])
    sim_sh = np.array([[sharpness(sim[a, H + r]) for r in range(R)] for a in range(n)])
    recon_sh = np.array([[sharpness(recon[a, H + r]) for r in range(R)] for a in range(n)])
    ctx_m = np.array([measure(sim[a, H - 1]) for a in range(n)])
    gt_disp = np.hypot(*(real_m[:, :, :2] - ctx_m[:, None, :2]).transpose(2, 0, 1))  # (n,R) block motion
    res = {"H": H, "fi": FI, "rounds": R, "n_anchors": int(n), "ok_block_px": OK_BLOCK_PX, "ok_angle_deg": OK_ANGLE_DEG,
           "blur_edge_width_ratio": BLUR_EW_RATIO, "moved_px": MOVED_PX, "gt_block_disp_from_anchor_px": [boot_ci(gt_disp[:, r]) for r in range(R)],
           "arms": {}}  # fmt: skip
    moved = gt_disp.max(1) > MOVED_PX
    grids = args.out / "grids"
    grids.mkdir(exist_ok=True)
    cache = {}
    for name, _steps, na in _arms(args):
        p = args.out / f"wm_{name}.npz"
        if not p.exists():
            print(f"skip arm {name}: {p} missing")
            continue
        imag = np.load(p)["imag"]
        na = imag.shape[0]
        me, sh = per_sample(imag)
        cache[name] = (me, sh, na)
        m = arm_metrics(imag, me, sh, real_m[:na], recon_m[:na], recon_sh[:na], sim_sh[:na], moved[:na])
        m["steps"] = int(np.load(p)["steps"])
        res["arms"][name] = m
        np.savez_compressed(args.out / f"measures_{name}.npz", measures=me, sharpness=sh)
        pick = [R // 4 - 1, R // 2 - 1, R - 1] if R >= 4 else list(range(R))
        widest = np.argsort(-_pos_sd(me[:, :, R - 1]))[:3]
        for a in sorted(set(range(min(args.n_grids, na))) | {int(i) for i in widest}):
            g = make_grid(
                imag[a],
                sim[a],
                recon[a],
                pick,
                f"{name} ({m['steps']} steps)  anchor {a}: same context + same actions, {imag.shape[1]} WM seeds",
            )
            cv2.imwrite(
                str(grids / f"{name}_anchor{a:02d}.jpg"),
                cv2.cvtColor(g, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            z = make_zoom(
                imag[a],
                sim[a],
                recon[a],
                R - 1,
                real_m[a, R - 1, 0] * 224 / 512,
                real_m[a, R - 1, 1] * 224 / 512,
            )
            cv2.imwrite(str(grids / f"{name}_anchor{a:02d}_zoom.png"), cv2.cvtColor(z, cv2.COLOR_RGB2BGR))
    if "main" in cache and "steps250" in cache:
        me30, sh30, _ = cache["main"]
        me250, sh250, na2 = cache["steps250"]
        me30, sh30 = me30[:na2], sh30[:na2]
        rb = np.repeat(real_m[:na2, None], me30.shape[1], 1)
        e30, e250 = errors(me30, rb), errors(me250, rb)

        def pos_sd(me):
            return np.nanmean(_pos_sd(me.transpose(0, 2, 1, 3)), 1)  # (na,)

        res["paired_250_vs_30"] = {
            "band_lapvar_ratio_to_recon": boot_ci(np.nanmean(sh250[..., 1] / recon_sh[:na2, None, :, 1], (1, 2)) - np.nanmean(sh30[..., 1] / recon_sh[:na2, None, :, 1], (1, 2))),
            "edge_width_ratio_to_recon": boot_ci(np.nanmean(sh250[..., 2] / recon_sh[:na2, None, :, 2], (1, 2)) - np.nanmean(sh30[..., 2] / recon_sh[:na2, None, :, 2], (1, 2))),
            "edge_width_px": boot_ci(np.nanmean(sh250[..., 2], (1, 2)) - np.nanmean(sh30[..., 2], (1, 2))),
            "block_err_px": boot_ci(np.nanmean(e250["block_px"], (1, 2)) - np.nanmean(e30["block_px"], (1, 2))),
            "block_pos_sd_px": boot_ci(pos_sd(me250) - pos_sd(me30)),
            "vanished_rate": boot_ci(e250["vanished"].mean((1, 2)) - e30["vanished"].mean((1, 2))),
        }  # fmt: skip
        for a in range(min(args.n_grids, na2)):
            ga = make_grid(
                np.load(args.out / "wm_main.npz")["imag"][a],
                sim[a],
                recon[a],
                [R - 1],
                f"anchor {a}: 30 steps (last round)",
            )
            gb = make_grid(
                np.load(args.out / "wm_steps250.npz")["imag"][a],
                sim[a],
                recon[a],
                [R - 1],
                f"anchor {a}: 250 steps (last round)",
            )
            cv2.imwrite(
                str(grids / f"steps30_vs_250_anchor{a:02d}.jpg"),
                cv2.cvtColor(np.concatenate([ga, gb], 0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
    det = args.out / "detcheck.json"
    if det.exists():
        res["detcheck"] = json.loads(det.read_text())
    (args.out / "summary.json").write_text(json.dumps(res, indent=1))
    (args.out / "summary.md").write_text(summary_md(res))
    print(summary_md(res))
    print(f"wrote {args.out / 'summary.json'}, {args.out / 'summary.md'}, {grids}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["real", "wm", "recon", "detcheck", "analyze", "all"])
    p.add_argument(
        "--out", type=Path, default=Path("~/datastorage/nanowm_pusht_aug/wm_seed_diversity").expanduser()
    )
    p.add_argument("--ckpt", type=Path, default=Path(COMMON["ckpt"]["base"]))
    p.add_argument(
        "--seeds", type=Path, default=DEFAULT_SEEDS, help="body-origin anchor states (save_seeds npz)"
    )
    p.add_argument("--n-anchors", type=int, default=20)
    p.add_argument(
        "--n-anchors-250", type=int, default=6, help="anchors in the 250-step control arm (0 = skip)"
    )
    p.add_argument("--n-wm-seeds", type=int, default=16)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--num-sampling-steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--wm-batch", type=int, default=int(COMMON["nanowm"]["wm_batch"]))
    p.add_argument("--wm-seed", type=int, default=0, help="worker --seed")
    p.add_argument("--torch-seed", type=int, default=0, help="policy seed for the real rollout")
    p.add_argument("--arm", nargs="*", default=None, help="wm stage: only these arms (main, steps250)")
    p.add_argument("--n-grids", type=int, default=6, help="anchors to draw grids for")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out = args.out.expanduser()
    stages = ["real", "wm", "recon", "detcheck", "analyze"] if args.stage == "all" else [args.stage]
    for st in stages:
        {
            "real": stage_real,
            "wm": stage_wm,
            "recon": stage_recon,
            "detcheck": stage_detcheck,
            "analyze": stage_analyze,
        }[st](args)


if __name__ == "__main__":
    main()
