#!/usr/bin/env python
"""T1.1b -- end-to-end calibration of the rejection-sampling ACCEPT FILTER (lerobot venv).

The filter's job is to predict, from an IMAGINED rollout alone, whether the executed action
stream is a REAL success (gym_pusht: coverage > 0.95 at any point -- `terminated = is_success
= coverage > self.success_threshold`, so success is a "ever reach" criterion, not a terminal one).

Two error sources stack:
  * estimator error -- now small after the T0.2 quadratic calibration (bias ~+0.005 vs sim)
  * world-model drift -- imagined coverage runs ~+0.044 high over 8 rounds; this now dominates

Guessing a threshold from either in isolation is wrong, so this measures the filter end to end:
run the closed loop with NO filter (`--accept-cov 0`), replay EVERY branch in real gym_pusht
from its post-lead-in anchor, and tabulate precision/recall of candidate accept rules against
the real outcome. Precision is what matters -- fine-tuning on false accepts reinforces exactly
the ~0.93-coverage stall the policy already has.

    uv run --extra training python examples/nanowm_pusht_aug/eval/proto_filter_roc.py \
        --branches examples/nanowm_pusht_aug/cl/all_roc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np

SUCCESS = 0.95


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branches", type=Path, required=True)
    p.add_argument("--fi", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("examples/nanowm_pusht_aug/diag/filter_roc.npz"))
    args = p.parse_args()

    files = sorted(args.branches.glob("*.npz"))
    if not files:
        raise SystemExit(f"no branches in {args.branches}")
    print(f"replaying {len(files)} branches in real gym_pusht ...")

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels",
        observation_width=224,
        observation_height=224,
        max_episode_steps=400,
    )
    u = env.unwrapped
    I_term, I_peak, I_rawterm, I_rawpeak = [], [], [], []
    R_peak_sim, R_term_sim = [], []
    for f in files:
        d = np.load(f)
        tgts = d["exec_abs_targets"]
        ic, icr = d["coverage"], d.get("coverage_raw", d["coverage"])
        st = d["anchor_state"]
        env.reset(seed=0)
        u.agent.position = tuple(float(x) for x in st[:2])
        u.block.angle = float(st[4])
        u.block.position = tuple(float(x) for x in st[2:4])
        u.block.velocity = (0.0, 0.0)
        u.block.angular_velocity = 0.0
        u.space.step(u.dt)
        curve = []
        for t in range(len(tgts)):
            env.step(np.asarray(tgts[t], dtype=np.float64))
            curve.append(u._get_coverage())  # simulator ground truth
        curve = np.array(curve)
        I_term.append(float(ic[-1]))
        I_peak.append(float(ic.max()))
        I_rawterm.append(float(icr[-1]))
        I_rawpeak.append(float(icr.max()))
        R_peak_sim.append(float(curve.max()))
        R_term_sim.append(float(curve[-1]))
    env.close()

    I_term, I_peak = np.array(I_term), np.array(I_peak)
    I_rawterm, I_rawpeak = np.array(I_rawterm), np.array(I_rawpeak)
    R_peak, R_term = np.array(R_peak_sim), np.array(R_term_sim)
    success = R_peak > SUCCESS  # gym_pusht's own success criterion

    n = len(files)
    print(
        f"\nbranches={n}   REAL successes (sim peak > {SUCCESS}): {success.sum()} ({success.mean() * 100:.1f}%)"
    )
    print(
        f"  real peak coverage: median {np.median(R_peak):.3f}  p75 {np.percentile(R_peak, 75):.3f} "
        f"p90 {np.percentile(R_peak, 90):.3f}  max {R_peak.max():.3f}"
    )
    print(f"  imagined vs real drift: mean(imag_peak - real_peak) = {np.mean(I_peak - R_peak):+.3f}")

    for nm, score in (
        ("imagined PEAK (calibrated)", I_peak),
        ("imagined TERMINAL (calibrated)", I_term),
        ("imagined PEAK (raw)", I_rawpeak),
        ("imagined TERMINAL (raw)", I_rawterm),
    ):
        print(f"\n=== accept rule: {nm} >= t ===")
        print(
            f"  {'t':>7s} {'kept':>6s} {'rate':>7s} {'precision':>10s} {'recall':>8s} {'real peak of kept':>18s}"
        )
        for t in (0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98):
            k = score >= t
            if k.sum() == 0:
                continue
            print(
                f"  {t:7.2f} {k.sum():6d} {k.mean():7.3f} {success[k].mean():10.3f} "
                f"{(success & k).sum() / max(success.sum(), 1):8.3f} {np.median(R_peak[k]):18.3f}"
            )
        if score.max() < 0.98:
            print(f"  (score caps at {score.max():.3f})")

    print("\nGuidance: pick the highest-precision rule that still keeps enough branches.")
    print("  A rule whose kept-set median real peak is <= ~0.93 is WORTHLESS -- that is where the")
    print("  policy already stalls, so those samples reinforce the failure mode.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        I_term=I_term,
        I_peak=I_peak,
        I_rawterm=I_rawterm,
        I_rawpeak=I_rawpeak,
        R_peak=R_peak,
        R_term=R_term,
        success=success,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
