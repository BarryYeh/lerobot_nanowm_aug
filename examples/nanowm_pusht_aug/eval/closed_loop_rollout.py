#!/usr/bin/env python
"""Closed-loop imagined rollout + rejection sampling, with CORRECT context/action alignment.

Drives the NanoWM worker (`imagined_rollout.py closed-loop`, nanowm venv) over .npz files
in a work dir; this side owns the policy, that side owns the world model.

The first closed-loop attempt accepted 0/128 on the strict-terminal criterion and caused the
project's original fail-fast. It seeded the worker with a SINGLE frame replicated to H context
slots and started the real actions at action group 0, so the first (H-1)*fi steps of every window
described a fictitious past and the real actions landed ~1.0 s early. Fixing the alignment moved
open-loop tracking from corr 0.347 to 0.958, and an offset sweep degrades monotonically (off0
0.984, off2 0.934, off5 0.412, off8 0.358, off11 0.259). Both scripts involved are gone; the
analysis is in diag/ROOT_CAUSE_BRIEF.md and the worker now REFUSES an under-length context
instead of padding it.

ALIGNMENT CONTRACT. NanoWM's DFoT call takes nf(=4) frames -- H(=3) context + 1 generated --
plus an action window of nf*fi(=20) sim steps reshaped to (nf, fi*2). Action group j drives
the transition STARTING AT FRAME j, so round r's window must span stream steps
[r*fi, r*fi + nf*fi), where the stream is indexed from the FIRST context frame.

  stream[0 .. (H-1)*fi)          real gym lead-in actions (policy rolled from the seed state)
  round r window                 stream[r*fi : r*fi+10]  ++  chunk[0:10]   (10 executed + 10 lookahead)
  after round r                  append chunk[0:fi] to the stream; advance the PD agent fi steps

Per round each branch re-plans on its own imagined frame, so this is a true closed loop.
Strict-terminal accept: imagined coverage[last] >= --accept-cov.

    uv run --extra training python examples/nanowm_pusht_aug/eval/closed_loop_rollout.py \
        --ckpt outputs/train/pushtnoise_clean/obs1_s100000/checkpoints/last/pretrained_model \
        --seeds examples/nanowm_pusht_aug/cl/seeds_pn_mid.npz --n-seeds 32 --k 4 --h-max 8 \
        --work examples/nanowm_pusht_aug/cl/work_cl_aligned \
        --out  examples/nanowm_pusht_aug/cl/accepted_aligned
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES

_PROJ = Path(__file__).resolve().parent.parent  # examples/nanowm_pusht_aug
sys.path.insert(0, str(_PROJ))
from datagen import action_convert as ac  # noqa: E402

NANOWM_PY = Path(__file__).resolve().parents[3] / "third_party/nano-world-model/.venv/bin/python"


def _wait(path: Path, timeout=1800.0, stop: Path | None = None):
    t0 = time.time()
    while not path.exists():
        if stop is not None and stop.exists():
            raise RuntimeError(f"worker stopped before {path.name}")
        if time.time() - t0 > timeout:
            raise TimeoutError(str(path))
        time.sleep(0.3)


def _save_atomic(path: Path, **a):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **a)
    tmp.rename(path)


def _to96(f224: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(f224).float().permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(96, 96), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


@torch.no_grad()
def sample_chunks(policy, pre, post, frames96: np.ndarray, k: int, dev: str) -> np.ndarray:
    """(M,96,96,3) u8 -> (M*k, horizon, 2) absolute-target chunks, k independent per frame.

    Goes through `diffusion.conditional_sample` directly: the public `generate_actions` slices
    to n_action_steps, and the window lookahead needs the full horizon. (Same call as
    the original closed-loop script, which was validated end-to-end.)
    """
    img = torch.from_numpy(frames96).float().permute(0, 3, 1, 2).div(255.0).to(dev)
    img = pre({"observation.image": img})["observation.image"]
    M = img.shape[0]
    x = img[:, None, None].repeat_interleave(k, dim=0)  # (M*k, n_obs=1, n_cam=1, 3, 96, 96)
    gcond = policy.diffusion._prepare_global_conditioning({OBS_IMAGES: x})
    norm_act = policy.diffusion.conditional_sample(M * k, global_cond=gcond)
    return post(norm_act).detach().cpu().numpy().astype(np.float64)


SIGMA_PX = ac.ACTION_SCALE * ac.NANOWM_A_STD  # ~20.8 px/step: std of NanoWM's training actions
AGENT_BOX = (22.0, 489.0)  # agent centre range that keeps a radius-15 agent inside the walls


def inject_noise(chunks, pos, vel, level: float, max_z: float, hold: int, rng) -> np.ndarray:
    """Local exploration in NanoWM sigma units -- the scheme wm_fidelity_ood.py (--inject sigma)
    validated against the simulator:

        rel_t = (a_t - agent_t) + level * SIGMA_PX * z,   z ~ N(0, I) redrawn every `hold` steps
        rel_t = rel_t * min(1, max_z / |rel_t / SIGMA_PX|)

    agent_t is tracked along the chunk with the same PD model the stream uses, so the cap bounds the
    relative action NanoWM actually conditions on. The fidelity sweeps set the ceiling: 3 sigma broke
    NanoWM within one round and beta >= 1 by round 4, so levels above ~1 sigma are out of bounds.
    """
    out = chunks.copy()
    bs, hz, _ = chunks.shape
    z = rng.standard_normal((bs, -(-hz // hold), 2))
    for b in range(bs):
        pp, vv = pos[b].copy(), vel[b].copy()
        for t in range(hz):
            rel = (chunks[b, t] - pp) + level * SIGMA_PX * z[b, t // hold]
            rel *= min(1.0, max_z / max(float(np.linalg.norm(rel / SIGMA_PX)), 1e-9))
            out[b, t] = np.clip(pp + rel, *AGENT_BOX)
            pp, vv = ac.pd_step(pp, vv, out[b, t])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--n-seeds", type=int, default=32)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--h-max", type=int, default=8)
    p.add_argument("--fi", type=int, default=5)
    p.add_argument("--nf", type=int, default=4)
    p.add_argument("--hist", type=int, default=3)
    p.add_argument("--accept-cov", type=float, default=0.95)
    p.add_argument(
        "--accept-mode",
        choices=["terminal", "peak", "pair"],
        default="terminal",
        help="pair = keep a branch iff >=1 training pair can pass the per-pair rule "
        "(max cov over rounds 1..6 >= accept-cov); exactly the branches the "
        "packer can use, so nothing is stored that would be discarded later",
    )
    p.add_argument("--num-sampling-steps", type=int, default=30)
    p.add_argument("--wm-batch", type=int, default=24)
    p.add_argument(
        "--inject-sigma",
        type=float,
        default=0.0,
        help="local action noise in NanoWM sigma units (0 = off, the original closed loop). "
        "Keep <= 1: the fidelity sweeps show NanoWM losing the block at higher levels",
    )
    p.add_argument("--max-z", type=float, default=3.0, help="cap on |relative action| in sigma units")
    p.add_argument("--noise-hold", type=int, default=5, help="steps each noise draw is held")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument(
        "--save-latents",
        action="store_true",
        help="value-model stage 2: ask the worker for its NanoWM latents and store per-branch `latents` "
        "(H+h_max,4,32,32) f16, `lead_acts` (lead,2), `seed_index` (row in --seeds, i.e. the "
        "reset(seed=9000+i) index) and `agent_state_hist` (h_max+1,4) = PD-tracked [pos,vel] at stream "
        "index lead + r*fi (r=0..h_max) in every saved branch npz",
    )
    p.add_argument("--work", type=Path, default=Path("examples/nanowm_pusht_aug/cl/work_cl_aligned"))
    p.add_argument("--out", type=Path, default=Path("examples/nanowm_pusht_aug/cl/accepted_aligned"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    work, out = args.work, args.out
    for d in (work, out):
        d.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*"):
        f.unlink()
    dev, H, fi, nf, k = args.device, args.hist, args.fi, args.nf, args.k
    lead = (H - 1) * fi  # 10 sim steps of real lead-in

    sd = np.load(args.seeds)
    states = sd["states"][: args.n_seeds].astype(np.float64)
    M = len(states)
    bs = M * k
    print(f"seeds={M} k={k} branches={bs} h_max={args.h_max} accept_cov={args.accept_cov}")

    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )

    # ---------- real gym lead-in: H context frames + the actions that produced them ----------
    env96 = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        observation_width=96,
        observation_height=96,
        max_episode_steps=lead + 40,
    )
    env224 = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels",
        observation_width=224,
        observation_height=224,
        max_episode_steps=lead + 40,
    )
    u96, u224 = env96.unwrapped, env224.unwrapped
    ctx = np.zeros((M, H, 224, 224, 3), np.uint8)
    lead_acts = np.zeros((M, lead, 2), np.float64)
    lead_pos = np.zeros((M, lead + 1, 2), np.float64)
    lead_vel = np.zeros((M, lead + 1, 2), np.float64)
    last96 = np.zeros((M, 96, 96, 3), np.uint8)
    anchor_state = np.zeros((M, 5), np.float64)  # gym state where the IMAGINED rollout starts
    for i in range(M):
        st = states[i]
        for e_, uu in ((env96, u96), (env224, u224)):
            e_.reset(seed=9000 + i)
            uu.agent.position = (float(st[0]), float(st[1]))
            uu.block.angle = float(st[4])
            uu.block.position = (float(st[2]), float(st[3]))
            uu.block.velocity = (0.0, 0.0)
            uu.block.angular_velocity = 0.0
            uu.space.step(uu.dt)
        policy.reset()
        obs = u96.get_obs()
        ci = 0
        for t in range(lead + 1):
            if t % fi == 0 and ci < H:
                ctx[i, ci] = np.asarray(u224.get_obs())
                ci += 1
            lead_pos[i, t] = np.asarray(u96.agent.position)
            lead_vel[i, t] = np.asarray(u96.agent.velocity)
            if t == lead:
                last96[i] = np.asarray(obs["pixels"])
                anchor_state[i] = [
                    u96.agent.position[0],
                    u96.agent.position[1],
                    u96.block.position[0],
                    u96.block.position[1],
                    u96.block.angle,
                ]
                break  # frame at step `lead` captured, do NOT step
            img = (
                torch.from_numpy(np.asarray(obs["pixels"])).permute(2, 0, 1).float().div(255.0)[None].to(dev)
            )
            with torch.no_grad():
                a = post(policy.select_action(pre({"observation.image": img})))
            a = a[0].cpu().numpy().astype(np.float64)
            lead_acts[i, t] = a
            for e_ in (env96, env224):
                e_.step(a)
            obs = u96.get_obs()
    env96.close()
    env224.close()
    print(f"lead-in done: {H} real context frames per seed, {lead} real actions per seed")

    # expand seed-level arrays to branch level
    ctx_b = np.repeat(ctx, k, axis=0)  # (bs,H,224,224,3)
    stream = np.repeat(lead_acts, k, axis=0)  # (bs,lead,2) grows by fi each round
    pos = np.repeat(lead_pos[:, lead], k, axis=0).copy()  # agent state at stream index `lead`
    vel = np.repeat(lead_vel[:, lead], k, axis=0).copy()
    apos = np.repeat(lead_pos[:, :lead], k, axis=0)  # (bs,lead,2) agent pos per stream step
    avel = np.repeat(lead_vel[:, :lead], k, axis=0)  # (bs,lead,2) agent vel per stream step
    sid = np.repeat(np.arange(M), k)
    cand = np.tile(np.arange(k), M)

    _save_atomic(work / "seeds_expanded.npz", frames_224=ctx_b)
    stop = work / "stop"
    wlog = open(work / "wm_worker.log", "w")  # noqa: SIM115 -- subprocess stdout, closed below
    wenv = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    extra_worker_args = ["--save-latents"] if args.save_latents else []
    proc = subprocess.Popen(
        [
            str(NANOWM_PY),
            str(_PROJ / "imagined_rollout.py"),
            "closed-loop",
            "--work",
            str(work),
            "--rounds",
            str(args.h_max),
            "--history",
            str(H),
            "--num-sampling-steps",
            str(args.num_sampling_steps),
            "--wm-batch",
            str(args.wm_batch),
            *extra_worker_args,
        ],
        stdout=wlog,
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        env=wenv,
    )

    cur96 = np.repeat(last96, k, axis=0)  # policy input for round 0 (real frame)
    noise_rng = np.random.default_rng(args.noise_seed)
    if args.inject_sigma > 0:
        print(
            f"action noise: {args.inject_sigma} sigma, max_z {args.max_z}, hold {args.noise_hold}", flush=True
        )
    cov_hist = np.zeros((bs, args.h_max), np.float32)
    cov_raw_hist = np.zeros((bs, args.h_max), np.float32)
    frames_hist = np.zeros((bs, args.h_max, 224, 224, 3), np.uint8)
    exec_stream = np.zeros((bs, args.h_max * fi, 2), np.float64)
    # PD-tracked [pos, vel] at stream index lead + r*fi; entry 0 is the (real) anchor state
    agent_state_hist = np.zeros((bs, args.h_max + 1, 4), np.float64)
    agent_state_hist[:, 0] = np.concatenate([pos, vel], axis=1)
    try:
        for r in range(args.h_max):
            # k independent chunks only at round 0; afterwards each branch re-plans on its own frame
            chunks = sample_chunks(
                policy, pre, post, cur96[::k] if r == 0 else cur96, k if r == 0 else 1, dev
            )  # (bs, horizon, 2)
            if args.inject_sigma > 0:
                chunks = inject_noise(
                    chunks, pos, vel, args.inject_sigma, args.max_z, args.noise_hold, noise_rng
                )
            need = nf * fi
            wins = np.zeros((bs, nf, fi * 2), np.float32)
            for b in range(bs):
                k0 = r * fi
                past = stream[b, k0 : k0 + (H - 1) * fi + fi]  # executed part, 10 entries
                look = chunks[b, : need - len(past)]  # lookahead from the chunk
                seg = np.concatenate([past, look], 0)[:need]
                # exact rel: targets minus the agent trajectory, bootstrapped from the agent
                # state recorded at stream index k0
                rel, _, _ = ac.abs_targets_to_rel(seg, apos[b, k0], avel[b, k0])
                wins[b] = ac.rel_to_nanowm(rel[:need])
            _save_atomic(work / f"chunks_{r}.npz", win=wins, live=np.ones(bs, bool))
            _wait(work / f"frames_{r + 1}.npz", stop=stop)
            fr = np.load(work / f"frames_{r + 1}.npz")
            cov_hist[:, r] = fr["cov"]
            if "cov_raw" in fr:
                cov_raw_hist[:, r] = fr["cov_raw"]
            frames_hist[:, r] = fr["frames"]
            print(
                f"round {r}: imagined cov mean={fr['cov'].mean():.3f} max={fr['cov'].max():.3f} "
                f">=0.9: {(fr['cov'] >= 0.9).sum()}/{bs}",
                flush=True,
            )

            # advance: execute the chunk's first fi targets, extend the stream and agent track
            ex = chunks[:, :fi]
            exec_stream[:, r * fi : (r + 1) * fi] = ex
            newpos = np.zeros((bs, fi, 2))
            newvel = np.zeros((bs, fi, 2))
            for b in range(bs):
                pp, vv = pos[b].copy(), vel[b].copy()
                for t in range(fi):
                    newpos[b, t], newvel[b, t] = pp, vv
                    pp, vv = ac.pd_step(pp, vv, ex[b, t])
                pos[b], vel[b] = pp, vv
            agent_state_hist[:, r + 1] = np.concatenate([pos, vel], axis=1)
            stream = np.concatenate([stream, ex], 1)
            apos = np.concatenate([apos, newpos], 1)
            avel = np.concatenate([avel, newvel], 1)
            cur96 = np.stack([_to96(fr["frames"][b]) for b in range(bs)])
        stop.touch()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            stop.touch()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        wlog.close()

    latents = None
    if args.save_latents:
        latents = np.load(work / "latents.npz")["latents"]  # (bs, H+h_max, 4, 32, 32) f16
        assert latents.shape[:2] == (bs, H + args.h_max), latents.shape

    term = cov_hist[:, -1]
    peak = cov_hist.max(1)
    if args.accept_mode == "terminal":
        acc = term >= args.accept_cov
    elif args.accept_mode == "peak":
        acc = peak >= args.accept_cov
    else:  # pair
        acc = cov_hist[:, 1 : max(2, args.h_max - 1)].max(1) >= args.accept_cov
    print("\n" + "=" * 72)
    print(f"branches={bs}  terminal cov: mean {term.mean():.3f} max {term.max():.3f}")
    print(f"peak cov:     mean {peak.mean():.3f} max {peak.max():.3f}")
    for thresh in (0.8, 0.9, 0.95):
        print(
            f"  terminal >= {thresh}: {(term >= thresh).sum():4d}/{bs}   peak >= {thresh}: {(peak >= thresh).sum():4d}/{bs}"
        )
    print(f"ACCEPTS [mode={args.accept_mode}] (>= {args.accept_cov}): {acc.sum()}/{bs}")
    print("=" * 72)

    n = 0
    for b in np.nonzero(acc)[0]:
        extra = {}
        if latents is not None:
            extra = {
                "latents": latents[b],
                "lead_acts": lead_acts[sid[b]],
                "seed_index": np.int64(sid[b]),
                "agent_state_hist": agent_state_hist[b],
            }
        np.savez(
            out / f"seed{sid[b]:04d}_cand{cand[b]}.npz",
            frames=frames_hist[b],
            exec_abs_targets=exec_stream[b],
            seed_state=states[sid[b]],
            anchor_state=anchor_state[sid[b]],
            coverage=cov_hist[b],
            coverage_raw=cov_raw_hist[b],
            meta=np.array([k, args.h_max, args.num_sampling_steps], np.int32),
            anchor_frame=ctx[sid[b], H - 1],  # REAL 224px frame at the anchor (exec stream index 0)
            inject_sigma=np.float32(args.inject_sigma),
            **extra,
        )
        n += 1
    np.savez(
        work / "cl_result.npz",
        cov_hist=cov_hist,
        cov_raw_hist=cov_raw_hist,
        term=term,
        peak=peak,
        sid=sid,
        cand=cand,
        exec_stream=exec_stream,
    )
    print(f"wrote {n} accepted rollouts to {out}")


if __name__ == "__main__":
    main()
