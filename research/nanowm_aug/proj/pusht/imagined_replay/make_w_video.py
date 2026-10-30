#!/usr/bin/env python
"""Video: real rollout vs NanoWM imagination for the last-mile W anchors (Img-W experiment), with coverage.

Per W anchor (179 in all; --n of them are shown, picked at fixed percentiles of imagination error, not by hand):
  left   REAL     simulator replay of the recorded actions (policy lead-in, then the executed W chunk):
                  frames at steps 0, 5, 10, 15, 20; image-estimated AND simulator coverage.
  right  IMAGINED steps 0 and 5 = the same real context frames; step 10 = F̂(10), the Img-W training frame (real
                  context, 5 hold steps added); steps 15 and 20 = the last-mile imagination of the chosen candidate
                  (rounds 1 and 2; the round-2 frame has the generated step-15 frame in its context).
Coverage on both panels is the image-based estimate `coverage_from_frame(...)['coverage_cal']` (same estimator),
so panels are directly comparable; the real panel also shows the simulator's own coverage.
A closing card gives the statistics over all 179 anchors.

    uv run --extra training python .../make_w_video.py --out ~/datastorage/nanowm_pusht_aug/imagined_replay/lastmile_w/video
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ir_common import lastmile_env, w_anchors

from nwaug_envs.pusht.coverage import coverage_from_frame

IR = Path.home() / "datastorage/nanowm_pusht_aug/imagined_replay/lastmile_w"
STEPS = (0, 5, 10, 15, 20)
SUCCESS_COV = 0.95
S, M = 336, 20  # panel image size, margin
W, H = 2 * M + 2 * S + 24, 600
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, FG, MUTED = (250, 250, 250), (35, 35, 35), (120, 120, 120)
GREEN, RED, BLUE, ORANGE = (60, 140, 30), (50, 50, 200), (200, 110, 20), (20, 100, 220)  # BGR


def est(frame: np.ndarray) -> tuple[float, np.ndarray]:
    c = coverage_from_frame(frame)
    return float(c["coverage_cal"]), np.asarray(c["pose_img"][:2], float)


def replay(i: int, seed_state, actions: np.ndarray) -> tuple[dict, np.ndarray]:
    """Real frames at STEPS and simulator coverage at every step 0..20 (the 20 actions are lead-in + W chunk)."""
    env = lastmile_env(i, seed_state)
    u = env.unwrapped
    frames, cov = {}, np.zeros(len(actions) + 1)
    for t in range(len(actions) + 1):
        cov[t] = u._get_coverage()
        if t in STEPS:
            frames[t] = np.asarray(u.get_obs()).copy()
        if t < len(actions):
            env.step(actions[t])
    env.close()
    return frames, cov


def collect() -> list[dict]:
    d = np.load(IR / "imagined.npz")
    f10 = dict(zip(d["names"].tolist(), d["frames_img"], strict=True))
    rows = []
    for f, z, sig, i, li in w_anchors():
        acts = np.concatenate([li["lead_acts"][i], z["exec_abs_targets"]], 0)
        real, sim_cov = replay(i, li["seed_states"][i], acts)
        assert np.array_equal(real[10], z["anchor_frame"]), (
            f.name
        )  # replay reproduces the stored anchor frame
        img = {0: real[0], 5: real[5], 10: f10[f.name], 15: z["frames"][0], 20: z["frames"][1]}
        e_real = {t: est(real[t]) for t in STEPS}
        e_img = {t: est(img[t]) for t in STEPS}
        ok = np.nonzero(sim_cov[11:] > SUCCESS_COV)[0]
        rows.append(
            {
                "name": f.name,
                "sigma": float(sig),
                "real": real,
                "img": img,
                "sim_cov": sim_cov,
                "est_real": {t: e_real[t][0] for t in STEPS},
                "est_img": {t: e_img[t][0] for t in STEPS},
                "cog_err": {t: float(np.linalg.norm(e_real[t][1] - e_img[t][1])) for t in (10, 15, 20)},
                "sim_success_step": int(ok[0]) + 11 if len(ok) else -1,
            }
        )
    return rows


def put(img, text, org, scale=0.5, color=FG, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def card(lines: list[tuple[str, float, tuple, int]]) -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)
    y = 60
    for text, scale, color, thick in lines:
        put(img, text, (M, y), scale, color, thick)
        y += int(34 * max(scale / 0.5, 1.0)) if text else 18
    return img


def rgb2bgr(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(x, (S, S), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)


STAGE = {
    0: ("real context", "real context (NanoWM input)"),
    5: ("real context", "real context (NanoWM input)"),
    10: ("anchor", "IMAGINED F^(10): Img-W frame"),
    15: ("rollout", "IMAGINED step 15: last-mile round 1"),
    20: ("rollout", "IMAGINED step 20: last-mile round 2"),
}


def frame_card(r: dict, k: int, n: int, pct: float, t: int) -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)
    put(
        img,
        f"Anchor {k}/{n}   {r['name'].removesuffix('.npz')}   (noise sigma {r['sigma']:g})",
        (M, 28),
        0.55,
        FG,
        1,
    )
    ss = r["sim_success_step"]
    sim_txt = (
        f"simulator: reaches success at step {ss}"
        if ss > 0
        else "simulator: NO success within the 10 executed steps"
    )
    put(
        img,
        f"imagination-error percentile P{pct:.0f}  |  {sim_txt}",
        (M, 52),
        0.45,
        GREEN if ss > 0 else RED,
        1,
    )
    xs = (M, M + S + 24)
    put(img, "REAL (simulator replay)", (xs[0], 84), 0.55, FG, 1)
    put(img, STAGE[t][1], (xs[1], 84), 0.5, ORANGE if t >= 10 else MUTED, 1)
    ys = 94
    for x, im, e in ((xs[0], r["real"][t], r["est_real"][t]), (xs[1], r["img"][t], r["est_img"][t])):
        img[ys : ys + S, x : x + S] = rgb2bgr(im)
        cv2.rectangle(
            img, (x - 2, ys - 2), (x + S + 1, ys + S + 1), GREEN if e >= SUCCESS_COV else (190, 190, 190), 3
        )
    yi = ys + S + 26
    sim = r["sim_cov"][t]
    put(
        img,
        f"image-est. coverage {r['est_real'][t]:.3f}",
        (xs[0], yi),
        0.55,
        GREEN if r["est_real"][t] >= SUCCESS_COV else FG,
        1,
    )
    put(
        img,
        f"simulator coverage  {sim:.3f}" + ("  SUCCESS" if sim > SUCCESS_COV else ""),
        (xs[0], yi + 24),
        0.5,
        FG,
        1,
    )
    put(
        img,
        f"image-est. coverage {r['est_img'][t]:.3f}",
        (xs[1], yi),
        0.55,
        GREEN if r["est_img"][t] >= SUCCESS_COV else FG,
        1,
    )
    if t >= 10:
        gap = r["est_img"][t] - r["est_real"][t]
        put(
            img,
            f"vs real: {gap:+.3f} coverage, block CoG {r['cog_err'][t]:.2f}px",
            (xs[1], yi + 24),
            0.5,
            FG,
            1,
        )
    else:
        put(img, "identical to the real panel", (xs[1], yi + 24), 0.5, MUTED, 1)
    # timeline
    y0, x0, x1 = H - 64, M + 30, W - M - 30
    cv2.line(img, (x0, y0), (x1, y0), (160, 160, 160), 2)
    for s in STEPS:
        x = int(x0 + (x1 - x0) * s / 20)
        on = s == t
        cv2.circle(img, (x, y0), 9 if on else 5, BLUE if on else (160, 160, 160), -1)
        put(img, f"step {s}", (x - 24, y0 + 28), 0.45, BLUE if on else MUTED, 1)
    put(img, "context", (x0 - 10, y0 - 18), 0.4, MUTED, 1)
    put(img, "recorded W chunk (10 steps)", (int(x0 + (x1 - x0) * 0.52), y0 - 18), 0.4, MUTED, 1)
    return img


def intro(n: int) -> np.ndarray:
    return card(
        [
            ("Real rollout vs NanoWM imagination", 0.85, FG, 2),
            ("last-mile W anchors (Img-W experiment)", 0.6, MUTED, 1),
            ("", 0, FG, 1),
            ("LEFT  real: simulator replay of the recorded actions", 0.5, FG, 1),
            ("       (policy lead-in, then the executed W chunk)", 0.5, MUTED, 1),
            ("RIGHT imagined: NanoWM frames from real context", 0.5, FG, 1),
            ("       step 10 = F^(10), the Img-W training frame", 0.5, MUTED, 1),
            ("       steps 15 / 20 = last-mile imagination of the chosen W", 0.5, MUTED, 1),
            ("       candidate (step 20 has the generated step-15 frame in context)", 0.5, MUTED, 1),
            ("", 0, FG, 1),
            ("Coverage on both panels = image-estimated, same estimator;", 0.5, FG, 1),
            ("the real panel also shows the simulator's coverage.", 0.5, FG, 1),
            ("Green frame = estimated coverage >= 0.95 (success level).", 0.5, FG, 1),
            ("", 0, FG, 1),
            ("Anchors are picked at fixed percentiles of imagination error,", 0.5, MUTED, 1),
            (f"not hand-picked. The last card summarises all {n} anchors.", 0.5, MUTED, 1),
        ]
    )


def summary(rows: list[dict]) -> tuple[np.ndarray, dict]:
    q = lambda x: float(np.median(x))  # noqa: E731
    gap = {t: np.array([abs(r["est_img"][t] - r["est_real"][t]) for r in rows]) for t in (10, 15, 20)}
    cog = {t: np.array([r["cog_err"][t] for r in rows]) for t in (10, 15, 20)}
    sel = np.array([max(r["est_img"][15], r["est_img"][20]) >= SUCCESS_COV for r in rows])
    sim_ok = np.array([r["sim_success_step"] > 0 for r in rows])
    real_est_ok = np.array([max(r["est_real"][15], r["est_real"][20]) >= SUCCESS_COV for r in rows])
    n = len(rows)
    st = {
        "n_anchors": n,
        "cog_err_median_px": {t: q(cog[t]) for t in cog},
        "abs_cov_gap_median": {t: q(gap[t]) for t in gap},
        "imagination_reaches_095_at_15_or_20": int(sel.sum()),
        "real_est_reaches_095_at_15_or_20": int(real_est_ok.sum()),
        "simulator_success_within_10_steps": int(sim_ok.sum()),
    }
    lines = [
        (f"All {n} W anchors", 0.85, FG, 2),
        ("", 0, FG, 1),
        ("Imagined vs real (median over anchors)", 0.55, FG, 1),
    ]
    for t in (10, 15, 20):
        lines.append(
            (
                f"  step {t}: block CoG error {st['cog_err_median_px'][t]:.2f}px,  |coverage gap| {st['abs_cov_gap_median'][t]:.3f}",
                0.5,
                MUTED,
                1,
            )
        )
    lines += [
        ("", 0, FG, 1),
        ("Did the rollout reach success level (coverage >= 0.95)?", 0.55, FG, 1),
        (
            f"  imagined, step 15 or 20 (the selection rule): {st['imagination_reaches_095_at_15_or_20']}/{n}",
            0.5,
            FG,
            1,
        ),
        (
            f"  real, same frames, image estimate:            {st['real_est_reaches_095_at_15_or_20']}/{n}",
            0.5,
            FG,
            1,
        ),
        (
            f"  real, simulator, any step within the 10:      {st['simulator_success_within_10_steps']}/{n}",
            0.5,
            FG,
            1,
        ),
    ]
    return card(lines), st


def encode(png_dir: Path, out: Path, fps: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(png_dir / "f_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        check=True,
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8, help="anchors shown")
    ap.add_argument("--hold", type=int, default=10, help="frames each step is held (fps 10 -> 1 s)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N anchors are replayed")
    a = ap.parse_args()
    rows = collect() if not a.limit else collect()[: a.limit]
    score = np.array([np.mean([r["cog_err"][t] for t in (10, 15, 20)]) for r in rows])
    order = np.argsort(score)
    pcts = np.linspace(5, 98, a.n)
    picks = [int(order[min(len(rows) - 1, round(p / 100 * (len(rows) - 1)))]) for p in pcts]
    png = a.out / "frames"
    shutil.rmtree(png, ignore_errors=True)
    png.mkdir(parents=True)
    cards, k = [(intro(len(rows)), 40)], 0
    for ai, (idx, p) in enumerate(zip(picks, pcts, strict=True), 1):
        cards += [(frame_card(rows[idx], ai, a.n, p, t), a.hold) for t in STEPS]
    sum_img, stats = summary(rows)
    cards.append((sum_img, 60))
    for im, rep in cards:
        for _ in range(rep):
            cv2.imwrite(str(png / f"f_{k:05d}.png"), im)
            k += 1
    video = a.out / "w_real_vs_imagined.mp4"
    encode(png, video, 10)
    shutil.rmtree(png)
    stats["shown"] = [
        {"anchor": rows[i]["name"], "percentile": float(p), "sim_success_step": rows[i]["sim_success_step"]}
        for i, p in zip(picks, pcts, strict=True)
    ]
    (a.out / "video_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"wrote {video} ({k} frames, {k / 10:.0f}s)")


if __name__ == "__main__":
    main()
