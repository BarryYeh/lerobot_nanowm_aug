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

Chunk cadence (`--replan-every-rounds R`, default 1 = re-plan every round, the original behaviour):
a chunk is sampled (and noised) only on rounds r % R == 0; round r then executes
`cur_chunks[:, off:off+fi]` with `off = (r % R) * fi` (see `chunk_slices`). R=2 matches the stage-1
10-step chunk cadence. Reproducibility (SAILOR v1): `--torch-seed` seeds torch before the lead-in and
again (seed+1) before round 0; every run writes `<work>/lead_in.npz` (see `save_lead_in`) and
`--lead-in <npz>` replays a saved lead-in's ACTIONS (no policy call, no torch RNG use) so several
runs of the same seeds share bit-identical anchor states / context frames (paired candidates).

    uv run --extra training python examples/nanowm_pusht_aug/eval/closed_loop_rollout.py \
        --ckpt outputs/train/pushtnoise_clean/obs1_s100000/checkpoints/last/pretrained_model \
        --seeds examples/nanowm_pusht_aug/cl/seeds_pn_mid.npz --n-seeds 32 --k 4 --h-max 8 \
        --work examples/nanowm_pusht_aug/cl/work_cl_aligned \
        --out  examples/nanowm_pusht_aug/cl/accepted_aligned
"""

from __future__ import annotations

import argparse
import hashlib
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
from lerobot.utils.constants import ACTION, OBS_IMAGES

_PROJ = Path(__file__).resolve().parent.parent  # examples/nanowm_pusht_aug
sys.path.insert(0, str(_PROJ))
from datagen import action_convert as ac  # noqa: E402
from nwaug.pusht_state import load_seeds, place_body, read_body  # noqa: E402

NANOWM_PY = Path(__file__).resolve().parents[3] / "third_party/nano-world-model/.venv/bin/python"


def _wait(path: Path, timeout=1800.0, stop: Path | None = None, proc=None):
    t0 = time.time()
    while not path.exists():
        if stop is not None and stop.exists():
            raise RuntimeError(f"worker stopped before {path.name}")
        if proc is not None and proc.poll() is not None:
            if path.exists():
                return
            log = path.parent / "wm_worker.log"
            tail = ""
            if log.exists():
                lines = log.read_text().splitlines()
                tail = "\n".join(lines[-20:])
            raise RuntimeError(
                f"NanoWM worker exited with code {proc.returncode} before writing {path.name}; "
                f"last lines of {log}:\n{tail}"
            )
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


def chunk_slices(r: int, every: int, fi: int, need: int, past_len: int) -> tuple[bool, int, slice, slice]:
    """Round-r bookkeeping of the chunk cadence (pure; `every` = `--replan-every-rounds` R).

    Returns `(sample_now, off, look, ex)`:
      sample_now  a new chunk is sampled (and noised) this round iff `r % R == 0`;
      off         offset of this round's executed steps inside the current chunk, `(r % R) * fi`;
      look        slice of the chunk used as lookahead in the NanoWM window, `[off, off+need-past_len)`;
      ex          slice of the chunk executed this round, `[off, off+fi)`.
    R=1 gives `off=0`, `look=[0, need-past_len)`, `ex=[0, fi)` for every r -- the original code."""
    assert every >= 1 and r >= 0, (r, every)
    off = (r % every) * fi
    return r % every == 0, off, slice(off, off + need - past_len), slice(off, off + fi)


def validate_cadence(h_max: int, every: int) -> None:
    """`--replan-every-rounds` must divide `--h-max`, else the last chunk would be cut short."""
    assert every >= 1 and h_max % every == 0, (
        f"h_max {h_max} must be a multiple of --replan-every-rounds {every}"
    )


def run_lead_in(
    env96, env224, states, act_fn, *, hist: int, fi: int, lead: int, reset_base: int = 9000
) -> dict:
    """Real gym lead-in for every seed: `hist` 224px context frames (every `fi` steps) plus the `lead`
    actions that produced them. `act_fn(i, t, pixels96) -> (2,) float64` supplies the action of seed
    `i` at lead-in step `t` (called with t = 0 first, so a policy can reset there); the loop stops at
    step `lead` WITHOUT stepping, so the last frame / `anchor_state` are the state the imagination
    starts from. Used both for the policy lead-in and for the `--lead-in` replay (act_fn = saved actions)."""
    u96, u224 = env96.unwrapped, env224.unwrapped
    M = len(states)
    ctx = np.zeros((M, hist, 224, 224, 3), np.uint8)
    lead_acts = np.zeros((M, lead, 2), np.float64)
    lead_pos = np.zeros((M, lead + 1, 2), np.float64)
    lead_vel = np.zeros((M, lead + 1, 2), np.float64)
    last96 = np.zeros((M, 96, 96, 3), np.uint8)
    anchor_state = np.zeros((M, 5), np.float64)  # gym state where the IMAGINED rollout starts
    for i in range(M):
        st = states[i]
        for e_, uu in ((env96, u96), (env224, u224)):
            e_.reset(seed=reset_base + i)
            place_body(uu, st)
        obs = u96.get_obs()
        ci = 0
        for t in range(lead + 1):
            if t % fi == 0 and ci < hist:
                ctx[i, ci] = np.asarray(u224.get_obs())
                ci += 1
            lead_pos[i, t] = np.asarray(u96.agent.position)
            lead_vel[i, t] = np.asarray(u96.agent.velocity)
            if t == lead:
                last96[i] = np.asarray(obs["pixels"])
                anchor_state[i] = read_body(u96)
                break  # frame at step `lead` captured, do NOT step
            a = np.asarray(act_fn(i, t, np.asarray(obs["pixels"])), np.float64)
            lead_acts[i, t] = a
            for e_ in (env96, env224):
                e_.step(a)
            obs = u96.get_obs()
    return {
        "ctx": ctx,
        "lead_acts": lead_acts,
        "lead_pos": lead_pos,
        "lead_vel": lead_vel,
        "last96": last96,
        "anchor_state": anchor_state,
    }


def _ctx_sha1(ctx: np.ndarray) -> np.ndarray:
    return np.array(
        [hashlib.sha1(np.ascontiguousarray(c).tobytes(), usedforsecurity=False).hexdigest() for c in ctx]
    )


def save_lead_in(
    path: Path, res: dict, states: np.ndarray, *, hist: int, fi: int, lead_na: int, torch_seed: int | None
) -> None:
    """Persist what is needed to reproduce a lead-in exactly: the actions (`lead_acts`), the body-origin
    seed states, the resulting `anchor_state` and `agent_state0` = [pos, vel] at the anchor, the agent
    track, and a sha1 per seed of its context frames (frames are re-rendered from the actions)."""
    lead = res["lead_acts"].shape[1]
    np.savez(
        path,
        lead_acts=res["lead_acts"],
        anchor_state=res["anchor_state"],
        agent_state0=np.concatenate([res["lead_pos"][:, lead], res["lead_vel"][:, lead]], axis=1),
        lead_pos=res["lead_pos"],
        lead_vel=res["lead_vel"],
        seed_states=np.asarray(states, np.float64),
        ctx_sha1=_ctx_sha1(res["ctx"]),
        fi=np.int64(fi),
        hist=np.int64(hist),
        lead_na=np.int64(lead_na),
        torch_seed=np.int64(-1 if torch_seed is None else torch_seed),
    )


def load_lead_in(path: Path, states: np.ndarray, *, hist: int, fi: int) -> dict:
    """Load a saved lead-in and check it belongs to these seeds / geometry."""
    d = np.load(path)
    lead = (hist - 1) * fi
    assert int(d["fi"]) == fi and int(d["hist"]) == hist, (int(d["fi"]), int(d["hist"]), fi, hist)
    assert d["lead_acts"].shape == (len(states), lead, 2), (d["lead_acts"].shape, len(states), lead)
    assert np.array_equal(d["seed_states"], np.asarray(states, np.float64)), (
        f"{path}: saved seed_states differ from --seeds[: n_seeds]"
    )
    return {k: d[k] for k in d.files}


def check_lead_in_replay(saved: dict, res: dict) -> None:
    """The replayed lead-in must reproduce the saved one BIT FOR BIT (anchor state, agent state at the
    anchor, context frames) -- this is what makes candidates of different runs share a decision point."""
    lead = res["lead_acts"].shape[1]
    assert np.array_equal(saved["lead_acts"], res["lead_acts"])
    assert np.array_equal(saved["anchor_state"], res["anchor_state"]), "replayed anchor_state differs"
    agent0 = np.concatenate([res["lead_pos"][:, lead], res["lead_vel"][:, lead]], axis=1)
    assert np.array_equal(saved["agent_state0"], agent0), "replayed agent state at the anchor differs"
    assert np.array_equal(saved["ctx_sha1"], _ctx_sha1(res["ctx"])), "replayed context frames differ"


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
        "--replan-every-rounds",
        type=int,
        default=1,
        help="sample (and noise) a new chunk every R rounds, execute it in fi-step slices (default 1 = "
        "every round, the original behaviour). Must divide --h-max; R>1 needs policy horizon >= "
        "(R-1)*fi + need - (H-1)*fi (asserted)",
    )
    p.add_argument(
        "--torch-seed",
        type=int,
        default=None,
        help="torch.manual_seed before the lead-in, and again (seed+1) before round 0 so that a replayed "
        "lead-in (no RNG use) and a policy lead-in leave the same RNG state. None = no seeding (old)",
    )
    p.add_argument(
        "--lead-na",
        type=int,
        default=None,
        help="set policy.config.n_action_steps for the lead-in (asserted via the action-queue length); "
        "None = ckpt default (8 for pusht_imgonly)",
    )
    p.add_argument(
        "--lead-in",
        type=Path,
        default=None,
        help="lead_in.npz of an earlier run on the SAME seeds: replay its actions instead of calling the "
        "policy; the replayed anchor_state / agent state / context frames must match it bit for bit",
    )
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

    states = load_seeds(args.seeds, "body")[: args.n_seeds]
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
    R = args.replan_every_rounds
    validate_cadence(args.h_max, R)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    def policy_act(i: int, t: int, pix: np.ndarray) -> np.ndarray:
        if t == 0:
            if args.lead_na is not None:
                policy.config.n_action_steps = args.lead_na
            policy.reset()
            if args.lead_na is not None:
                assert policy._queues[ACTION].maxlen == args.lead_na, policy._queues[ACTION].maxlen
        img = torch.from_numpy(pix).permute(2, 0, 1).float().div(255.0)[None].to(dev)
        with torch.no_grad():
            a = post(policy.select_action(pre({"observation.image": img})))
        return a[0].cpu().numpy().astype(np.float64)

    saved_lead = None
    if args.lead_in is not None:
        saved_lead = load_lead_in(args.lead_in, states, hist=H, fi=fi)
        if args.lead_na is not None:
            assert int(saved_lead["lead_na"]) == args.lead_na, (int(saved_lead["lead_na"]), args.lead_na)
        act_fn = lambda i, t, pix: saved_lead["lead_acts"][i, t]  # noqa: E731 -- replay, no policy call
    else:
        act_fn = policy_act
    res = run_lead_in(env96, env224, states, act_fn, hist=H, fi=fi, lead=lead)
    if saved_lead is not None:
        check_lead_in_replay(saved_lead, res)
        print(
            f"lead-in REPLAYED from {args.lead_in} (anchor_state / agent state / context frames bit-identical)"
        )
    ctx, lead_acts, lead_pos, lead_vel = res["ctx"], res["lead_acts"], res["lead_pos"], res["lead_vel"]
    last96, anchor_state = res["last96"], res["anchor_state"]
    lead_na = (
        int(saved_lead["lead_na"])
        if saved_lead is not None
        else int(args.lead_na if args.lead_na is not None else policy.config.n_action_steps)
    )
    save_lead_in(
        work / "lead_in.npz", res, states, hist=H, fi=fi, lead_na=lead_na, torch_seed=args.torch_seed
    )
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
            need = nf * fi
            sample_now, off, look_sl, ex_sl = chunk_slices(r, R, fi, need, (H - 1) * fi)
            if sample_now:
                # k independent chunks only at round 0; afterwards each branch re-plans on its own frame
                if r == 0 and args.torch_seed is not None:
                    torch.manual_seed(
                        args.torch_seed + 1
                    )  # same state whether the lead-in used the RNG or not
                cur_chunks = sample_chunks(
                    policy, pre, post, cur96[::k] if r == 0 else cur96, k if r == 0 else 1, dev
                )  # (bs, horizon, 2)
                if r == 0:
                    chunk0_clean = cur_chunks.copy()  # un-noised round-0 chunks (paired-candidate check)
                if args.inject_sigma > 0:
                    cur_chunks = inject_noise(
                        cur_chunks, pos, vel, args.inject_sigma, args.max_z, args.noise_hold, noise_rng
                    )
            horizon = cur_chunks.shape[1]
            assert horizon >= look_sl.stop, (
                f"policy horizon {horizon} < {look_sl.stop} needed by round {r} (replan_every_rounds={R})"
            )
            wins = np.zeros((bs, nf, fi * 2), np.float32)
            for b in range(bs):
                k0 = r * fi
                past = stream[b, k0 : k0 + (H - 1) * fi + fi]  # executed part, 10 entries
                assert len(past) == (H - 1) * fi, (len(past), (H - 1) * fi)
                look = cur_chunks[b, look_sl]  # lookahead from the chunk
                seg = np.concatenate([past, look], 0)[:need]
                # exact rel: targets minus the agent trajectory, bootstrapped from the agent
                # state recorded at stream index k0
                rel, _, _ = ac.abs_targets_to_rel(seg, apos[b, k0], avel[b, k0])
                wins[b] = ac.rel_to_nanowm(rel[:need])
            _save_atomic(work / f"chunks_{r}.npz", win=wins, live=np.ones(bs, bool))
            _wait(work / f"frames_{r + 1}.npz", stop=stop, proc=proc)
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
            ex = cur_chunks[:, ex_sl]
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
            # cadence / reproducibility metadata (the labeler asserts the ones it relies on)
            fi=np.int64(fi),
            hist=np.int64(H),
            replan_every_rounds=np.int64(R),
            horizon=np.int64(cur_chunks.shape[1]),
            lead_na=np.int64(lead_na),
            torch_seed=np.int64(-1 if args.torch_seed is None else args.torch_seed),
            noise_hold=np.int64(args.noise_hold),
            chunk0_clean=chunk0_clean[b],  # round-0 chunk BEFORE noise (horizon, 2): paired-candidate check
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
