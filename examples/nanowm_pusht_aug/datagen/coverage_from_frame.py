#!/usr/bin/env python
"""Coverage + block pose from a rendered / imagined PushT RGB frame (Task 2).

Method (locked with the user): color-mask + moments.
  - block mask: slate-family chroma threshold (renderer- and VAE-tolerant) -> largest CC
  - block pose (image-space centroid + principal-axis angle) from moments -- diagnostic only
  - coverage = |block_mask & goal_mask| / |goal_mask|

The goal never moves (pose (256,256,pi/4)). Rasterizing it analytically from the pymunk
poly vertices is fiddly (pymunk recenters body-local coords on the centre of gravity), so
`goal_mask` is built ONCE from a real gym_pusht render (threshold the LightGreen fill) and
cached to `goal_mask_<size>.npy` next to this file. That cache is then usable from
the nanowm conda env too (which cannot import gym_pusht).

Runs in the lerobot venv for the cache build + sanity check; the cached .npy + `block_mask`
+ `coverage_from_frame` work in either venv.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
T_OUTLINE_RGB = (119, 136, 153)  # "LightSlateGray"  (pymunk poly outline = shape.color)
T_FILL_RGB = (143, 163, 184)  # 1.2 * outline      (pymunk poly fill = light_color(color))


def block_mask(rgb: np.ndarray) -> np.ndarray:
    """Slate-family T pixels -> cleaned largest-component binary mask.

    Chroma rule: slate is weakly blue-biased and mid-bright. Rejects white (b-r~0),
    grays, the LightGreen goal, and the saturated RoyalBlue agent (b-r~160). Unions
    tight RGB-distance hits to the exact outline/fill anchors for crisp renders.
    """
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    chroma = (b - r > 10) & (b - r < 95) & (b >= g - 5) & (r > 55) & (r < 205)
    do = np.abs(a - np.array(T_OUTLINE_RGB, np.int16)).sum(2) < 45
    df = np.abs(a - np.array(T_FILL_RGB, np.int16)).sum(2) < 55
    m = (chroma | do | df).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == biggest).astype(np.uint8)


def pose_from_mask(mask: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, theta) in IMAGE pixels / radians (image-space, y-down). Diagnostic only:
    theta is the T long axis, stem-vs-crossbar disambiguated by perpendicular spread."""
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] < 1:
        return (float("nan"), float("nan"), float("nan"))
    px, py = M["m10"] / M["m00"], M["m01"] / M["m00"]
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    ang = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    ys, xs = np.nonzero(mask)
    d = np.stack([xs - px, ys - py], 1).astype(np.float64)
    axis = np.array([math.cos(ang), math.sin(ang)])
    proj, perp = d @ axis, d @ np.array([-axis[1], axis[0]])
    lo = perp[proj < 0].std() if (proj < 0).any() else 0.0
    hi = perp[proj > 0].std() if (proj > 0).any() else 0.0
    if lo > hi:
        ang += math.pi
    return (px, py, ang)


_GOAL_CACHE: dict[int, np.ndarray] = {}


def goal_mask(size: int) -> np.ndarray:
    if size in _GOAL_CACHE:
        return _GOAL_CACHE[size]
    p = _HERE / f"goal_mask_{size}.npy"
    if p.exists():
        m = np.load(p).astype(np.uint8)
    else:
        m = build_goal_mask(size)
        np.save(p, m)
    _GOAL_CACHE[size] = m
    return m


def build_goal_mask(size: int) -> np.ndarray:
    """One-off: render gym_pusht with the block parked away, threshold LightGreen."""
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=size, observation_height=size)
    u = env.unwrapped
    env.reset(seed=0)
    u.block.position = (55.0, 55.0)
    u.block.angle = 0.0
    u.agent.position = (460.0, 460.0)
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)
    rgb = np.asarray(u.get_obs()).astype(np.int16)
    env.close()
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    m = ((g - r > 35) & (g - b > 35)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    return m


# Quadratic calibration of the raw mask-intersection coverage against the simulator, fit on 900
# near-goal gym renders at 224 (proto_calib_estimator.py). The raw estimate runs ~+0.04 high
# across most of the range because the thresholded block mask is slightly fat; the excess is
# clipped as the block saturates the goal, so the bias is curved, not a constant offset --
# a morphological erosion overcorrects badly (-0.10 at cov>=0.8) and is NOT the right fix.
#
#   accept band (true cov >= 0.8):  MAE 0.0279 -> 0.0120,  bias +0.0275 -> -0.0010
#   all bands:                      MAE 0.0399 -> 0.0107
#
# NOTE: fit on CRISP gym renders. VAE-decoded imagined frames are blurrier and may carry an
# additional bias on top of this; see proto_calib_vae_blur.py.
_COVERAGE_CALIB = (0.19652, 0.77843, 0.00441)


def calibrate_coverage(raw: float) -> float:
    """Map a raw mask-intersection coverage onto the simulator's coverage scale."""
    a, b, c = _COVERAGE_CALIB
    return float(np.clip(a * raw * raw + b * raw + c, 0.0, 1.0))


def coverage_from_frame(rgb: np.ndarray) -> dict:
    """rgb (H,W,3) uint8 -> dict(coverage, coverage_cal, pose_img, block_area_frac).

    `coverage` is the raw mask ratio (unchanged, so older diagnostics stay comparable);
    `coverage_cal` is the calibrated estimate and is what any accept/reject filter should use.
    """
    size = rgb.shape[0]
    bm = block_mask(np.asarray(rgb))
    gm = goal_mask(size)
    ga = int(gm.sum())
    cov = float((bm & gm).sum() / ga) if ga else 0.0
    return {
        "coverage": cov,
        "coverage_cal": calibrate_coverage(cov),
        "pose_img": pose_from_mask(bm),
        "block_area_frac": float(bm.mean()),
    }


# --------------------------------------------------------------------------------------
def _sanity(n: int = 150, size: int = 224) -> None:
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    gm = build_goal_mask(size)
    np.save(_HERE / f"goal_mask_{size}.npy", gm)
    print(f"built goal_mask {size}: {int(gm.sum())} px -> goal_mask_{size}.npy")

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=size, observation_height=size)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    errs, rows = [], []
    for i in range(n):
        st = np.array(
            [
                rng.integers(60, 450),
                rng.integers(60, 450),
                rng.integers(120, 400),
                rng.integers(120, 400),
                rng.uniform(-np.pi, np.pi),
            ]
        )
        env.reset(seed=int(i))
        u.agent.position = (float(st[0]), float(st[1]))
        u.block.angle = float(st[4])
        u.block.position = (float(st[2]), float(st[3]))
        u.block.velocity = (0.0, 0.0)
        u.block.angular_velocity = 0.0
        u.space.step(u.dt)
        rgb = np.asarray(u.get_obs())
        true_cov = u._get_coverage()
        r = coverage_from_frame(rgb)
        e = abs(r["coverage"] - true_cov)
        errs.append(e)
        rows.append((i, round(true_cov, 3), round(r["coverage"], 3), round(r["block_area_frac"], 4)))
    env.close()
    errs = np.array(errs)
    print(
        f"n={n}  coverage MAE={errs.mean():.4f}  median={np.median(errs):.4f}  "
        f"p90={np.percentile(errs, 90):.4f}  max={errs.max():.4f}  frac(err<0.05)={(errs < 0.05).mean():.2f}"
    )
    hi = np.array([r for r in rows if r[1] >= 0.7])
    if len(hi):
        he = np.abs(hi[:, 1] - hi[:, 2])
        print(
            f"  on true_cov>=0.7 (n={len(hi)}): MAE={he.mean():.4f}  frac(|err|<0.1)={(he < 0.1).mean():.2f}"
        )
    print("worst:")
    for w in sorted(rows, key=lambda r: -abs(r[1] - r[2]))[:8]:
        print("  ", w)


if __name__ == "__main__":
    _sanity()
