#!/usr/bin/env python
"""Method 1 (imagined-frame harm test) data prep: pick pairs, realize, self-label, write packable npz.

Criterion: ~/datastorage/nanowm_pusht_aug/lastmile/method1/criteria.md (fixed before this script ran).

  pick     121 of the 179 accepted last-mile starts (rng 970000), one random candidate each -> <out>/src symlinks
  (realize_branches.py --src <out>/src --dst <out>/realized gives the simulator frame at the same step)
  label    base policy samples a 16-step chunk on each 96px frame; img and real use the same torch seed and batch
           order (common random numbers). A second draw (seed + 1) gives each frame's sampling-noise floor.
           Writes <out>/pairs_img_self / pairs_real_self (npz in pack_closed_loop.py's anchor layout) and
           <out>/offline.json (action gap imagined vs real / noise, image coverage imagined vs simulator).
  label_r2 round 2 (criterion: method1_r2/criteria.md): the same branches' frames[1] (step 10), labelled as above.
           Writes <out>/pairs_{img,real}_r12 = method 1's 121 round-1 pairs (symlinks) + 121 round-2 pairs, and
           <out>/offline.json with the same numbers per round.

    uv run --extra training python research/nanowm_aug/proj/pusht/method1/method1_prep.py pick --out <dir>
    uv run --extra training python research/nanowm_aug/proj/pusht/method1/method1_prep.py label --out <dir> --ckpt <base>
    uv run --extra training python research/nanowm_aug/proj/pusht/method1/method1_prep.py label_r2 --out <r2 dir> \
        --ckpt <base> --m1 <method1 dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/nanowm_aug

LASTMILE = Path.home() / "datastorage/nanowm_pusht_aug/lastmile"
N_PAIRS = 121  # round(24002 * 0.005 / 0.995): 0.5% of the 24002 sampleable real frames at drop 8
PICK_SEED = 970000
LABEL_SEED = 980000
N_GAP = 10  # actions compared in the offline gap (= n_action_steps at eval)


def pick(out: Path) -> None:
    sel = json.loads((LASTMILE / "sel/selection.json").read_text())
    acc = sorted(r["start"] for r in sel["rows"] if r["accepted"])
    assert len(acc) == 179, len(acc)
    rng = np.random.default_rng(PICK_SEED)
    starts = sorted(rng.choice(acc, N_PAIRS, replace=False).tolist())
    src = out / "src"
    src.mkdir(parents=True, exist_ok=False)
    rows = []
    for s in starts:
        cands = sorted((LASTMILE / "imag").glob(f"sigma*/branches/seed{s:04d}_cand*.npz"))
        assert len(cands) == 12, (s, len(cands))
        f = cands[int(rng.integers(len(cands)))]
        name = f"{f.parent.parent.name}_{f.name}"
        (src / name).symlink_to(f.resolve())
        rows.append({"start": int(s), "branch": name})
    (out / "pick.json").write_text(json.dumps({"seed": PICK_SEED, "n": len(rows), "rows": rows}, indent=1))
    print(f"picked {len(rows)} pairs -> {src}")


def label(out: Path, ckpt: Path) -> None:
    import torch

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from nwaug.datagen.pack_closed_loop import to96
    from nwaug.wm.closed_loop_rollout import sample_chunks
    from nwaug_envs.pusht.coverage import coverage_from_frame

    rows = json.loads((out / "pick.json").read_text())["rows"]
    dev = "cuda"
    pol = DiffusionPolicy.from_pretrained(ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        pol.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": dev}}
    )
    img224, real224, meta = [], [], []
    for r in rows:
        zi = np.load(out / "src" / r["branch"])
        zr = np.load(out / "realized" / r["branch"])
        assert str(zr["frames_source"]) == "sim", r["branch"]
        assert np.array_equal(zi["exec_abs_targets"], zr["exec_abs_targets"]), r["branch"]
        img224.append(zi["frames"][0])
        real224.append(zr["frames"][0])
        meta.append(zi)
    frames = {"img": np.stack(img224), "real": np.stack(real224)}
    f96 = {k: np.stack([to96(f) for f in v]) for k, v in frames.items()}

    def draw(arm: str, seed: int) -> np.ndarray:
        torch.manual_seed(seed)  # same seed + same batch order for both arms = common random numbers
        with torch.no_grad():
            return sample_chunks(pol, pre, post, f96[arm], 1, dev)

    ch = {(arm, s): draw(arm, s) for arm in ("img", "real") for s in (LABEL_SEED, LABEL_SEED + 1)}
    gap = lambda a, b: np.linalg.norm(a[:, :N_GAP] - b[:, :N_GAP], axis=-1).mean(1)  # noqa: E731
    g_img_real = gap(ch["img", LABEL_SEED], ch["real", LABEL_SEED])
    noise_real = gap(ch["real", LABEL_SEED], ch["real", LABEL_SEED + 1])
    noise_img = gap(ch["img", LABEL_SEED], ch["img", LABEL_SEED + 1])
    cov = {k: np.array([coverage_from_frame(f)["coverage_cal"] for f in v]) for k, v in frames.items()}

    for arm in ("img", "real"):
        d = out / f"pairs_{arm}_self"
        d.mkdir(parents=True, exist_ok=False)
        for i, (r, z) in enumerate(zip(rows, meta, strict=True)):
            np.savez(
                d / f"pair{i:03d}_{r['branch']}",
                frames=frames[arm][i][None],  # 1 round -> pack_closed_loop emits only the anchor pair
                anchor_frame=frames[arm][i],
                exec_abs_targets=ch[arm, LABEL_SEED][i],  # (16, 2): the full horizon, no padding needed
                coverage=np.array([cov[arm][i]], np.float32),
                anchor_state=z[
                    "anchor_state"
                ],  # only feeds the packer's hold-pad track; unused at 16 actions
                inject_sigma=np.float32(0.0),
            )
    res = {
        "n": len(rows),
        "action_gap_px": {
            "img_vs_real": float(g_img_real.mean()),
            "noise_real": float(noise_real.mean()),
            "noise_img": float(noise_img.mean()),
            "ratio_img_real_over_noise_real": float(g_img_real.mean() / noise_real.mean()),
        },
        "coverage_cal": {
            "img_mean": float(cov["img"].mean()),
            "real_mean": float(cov["real"].mean()),
            "img_minus_real_mean": float((cov["img"] - cov["real"]).mean()),
            "abs_err_mean": float(np.abs(cov["img"] - cov["real"]).mean()),
        },
        "per_pair": [
            {
                "branch": r["branch"],
                "gap": float(g),
                "noise_real": float(n),
                "cov_img": float(ci),
                "cov_real": float(cr),
            }
            for r, g, n, ci, cr in zip(rows, g_img_real, noise_real, cov["img"], cov["real"], strict=True)
        ],
    }
    (out / "offline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_pair"}, indent=1))


def round_summary(gap: np.ndarray, noise: np.ndarray, cov_img: np.ndarray, cov_real: np.ndarray, cog) -> dict:
    """Offline numbers for one round: action gap vs noise (mean and median), coverage, block CoG error."""
    return {
        "action_gap_px": {
            "img_vs_real_mean": float(gap.mean()),
            "noise_real_mean": float(noise.mean()),
            "ratio_mean": float(gap.mean() / noise.mean()),
            "img_vs_real_median": float(np.median(gap)),
            "noise_real_median": float(np.median(noise)),
            "ratio_median": float(np.median(gap) / np.median(noise)),
            "n_gap_gt_noise": int((gap > noise).sum()),
        },
        "coverage_cal": {
            "img_mean": float(cov_img.mean()),
            "real_mean": float(cov_real.mean()),
            "img_minus_real_mean": float((cov_img - cov_real).mean()),
            "abs_err_mean": float(np.abs(cov_img - cov_real).mean()),
        },
        "cog_err_px": {"median": float(np.median(cog)), "p90": float(np.percentile(cog, 90))},
    }


def label_r2(out: Path, ckpt: Path, m1: Path) -> None:
    """Round 2: method 1's 121 round-1 pairs (symlinked) + 121 new self-labelled pairs on each branch's frames[1]."""
    import torch

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from nwaug.datagen.pack_closed_loop import to96
    from nwaug.wm.closed_loop_rollout import sample_chunks
    from nwaug_envs.pusht.coverage import coverage_from_frame

    rows = json.loads((m1 / "pick.json").read_text())["rows"]
    for d in ("src", "realized"):
        if not (out / d).exists():
            (out / d).symlink_to((m1 / d).resolve())
    cog = {b["branch"]: b["cog_err_px_per_round"] for b in json.loads((m1 / "realized/realize.json").read_text())["branches"]}
    dev = "cuda"
    pol = DiffusionPolicy.from_pretrained(ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        pol.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": dev}}
    )
    img224, real224, meta = [], [], []
    for r in rows:
        zi = np.load(out / "src" / r["branch"])
        zr = np.load(out / "realized" / r["branch"])
        assert str(zr["frames_source"]) == "sim", r["branch"]
        assert np.array_equal(zi["exec_abs_targets"], zr["exec_abs_targets"]), r["branch"]
        assert len(zi["frames"]) == len(zr["frames"]) == 2, r["branch"]
        img224.append(zi["frames"][1])
        real224.append(zr["frames"][1])
        meta.append(zi)
    frames = {"img": np.stack(img224), "real": np.stack(real224)}
    f96 = {k: np.stack([to96(f) for f in v]) for k, v in frames.items()}

    def draw(arm: str, seed: int) -> np.ndarray:
        torch.manual_seed(seed)  # same seed + same batch order for both arms = common random numbers
        with torch.no_grad():
            return sample_chunks(pol, pre, post, f96[arm], 1, dev)

    ch = {(arm, s): draw(arm, s) for arm in ("img", "real") for s in (LABEL_SEED, LABEL_SEED + 1)}
    gap = lambda a, b: np.linalg.norm(a[:, :N_GAP] - b[:, :N_GAP], axis=-1).mean(1)  # noqa: E731
    g_img_real = gap(ch["img", LABEL_SEED], ch["real", LABEL_SEED])
    noise_real = gap(ch["real", LABEL_SEED], ch["real", LABEL_SEED + 1])
    noise_img = gap(ch["img", LABEL_SEED], ch["img", LABEL_SEED + 1])
    cov = {k: np.array([coverage_from_frame(f)["coverage_cal"] for f in v]) for k, v in frames.items()}

    n1 = len(rows)
    for arm in ("img", "real"):
        d = out / f"pairs_{arm}_r12"
        d.mkdir(parents=True, exist_ok=False)
        r1 = sorted((m1 / f"pairs_{arm}_self").glob("*.npz"))
        assert len(r1) == n1, (arm, len(r1))
        for f in r1:  # round 1: method 1's pairs as they are
            (d / f.name).symlink_to(f.resolve())
        for i, (r, z) in enumerate(zip(rows, meta, strict=True)):
            np.savez(
                d / f"pair{n1 + i:03d}_r2_{r['branch']}",
                frames=frames[arm][i][None],  # 1 frame -> pack_closed_loop emits only the anchor pair
                anchor_frame=frames[arm][i],
                exec_abs_targets=ch[arm, LABEL_SEED][i],  # (16, 2): the full horizon, no padding needed
                coverage=np.array([cov[arm][i]], np.float32),
                anchor_state=z["anchor_state"],  # only feeds the packer's hold-pad track; unused at 16 actions
                inject_sigma=np.float32(0.0),
            )

    p1 = json.loads((m1 / "offline.json").read_text())["per_pair"]
    assert [p["branch"] for p in p1] == [r["branch"] for r in rows]
    a = lambda k: np.array([p[k] for p in p1])  # noqa: E731
    res = {
        "n_per_round": n1,
        "round1": round_summary(
            a("gap"), a("noise_real"), a("cov_img"), a("cov_real"), [cog[r["branch"]][0] for r in rows]
        ),
        "round2": round_summary(
            g_img_real, noise_real, cov["img"], cov["real"], [cog[r["branch"]][1] for r in rows]
        ),
        "per_pair_round2": [
            {
                "branch": r["branch"],
                "gap": float(g),
                "noise_real": float(n),
                "noise_img": float(ni),
                "cov_img": float(ci),
                "cov_real": float(cr),
                "cog_err_px": float(cog[r["branch"]][1]),
            }
            for r, g, n, ni, ci, cr in zip(
                rows, g_img_real, noise_real, noise_img, cov["img"], cov["real"], strict=True
            )
        ],
    }
    res["round2"]["action_gap_px"]["noise_img_mean"] = float(noise_img.mean())
    (out / "offline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_pair_round2"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pick", "label", "label_r2"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--m1", type=Path, default=LASTMILE / "method1", help="method 1 dir (label_r2 only)")
    a = ap.parse_args()
    if a.cmd == "pick":
        pick(a.out)
    else:
        assert a.ckpt is not None, f"--ckpt is required for {a.cmd}"
        if a.cmd == "label":
            label(a.out, a.ckpt)
        else:
            label_r2(a.out, a.ckpt, a.m1)


if __name__ == "__main__":
    main()
