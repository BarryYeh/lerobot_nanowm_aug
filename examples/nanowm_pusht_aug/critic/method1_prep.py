#!/usr/bin/env python
"""Method 1 (imagined-frame harm test) data prep: pick pairs, realize, self-label, write packable npz.

Criterion: ~/datastorage/nanowm_pusht_aug/lastmile/method1/criteria.md (fixed before this script ran).

  pick     121 of the 179 accepted last-mile starts (rng 970000), one random candidate each -> <out>/src symlinks
  (realize_branches.py --src <out>/src --dst <out>/realized gives the simulator frame at the same step)
  label    base policy samples a 16-step chunk on each 96px frame; img and real use the same torch seed and batch
           order (common random numbers). A second draw (seed + 1) gives each frame's sampling-noise floor.
           Writes <out>/pairs_img_self / pairs_real_self (npz in pack_closed_loop.py's anchor layout) and
           <out>/offline.json (action gap imagined vs real / noise, image coverage imagined vs simulator).

    uv run --extra training python examples/nanowm_pusht_aug/critic/method1_prep.py pick --out <dir>
    uv run --extra training python examples/nanowm_pusht_aug/critic/method1_prep.py label --out <dir> --ckpt <base>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent), str(_HERE.parent / "datagen"), str(_HERE.parent / "eval")]

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
    from closed_loop_rollout import sample_chunks
    from coverage_from_frame import coverage_from_frame
    from pack_closed_loop import to96

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pick", "label"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path)
    a = ap.parse_args()
    if a.cmd == "pick":
        pick(a.out)
    else:
        assert a.ckpt is not None, "--ckpt is required for label"
        label(a.out, a.ckpt)


if __name__ == "__main__":
    main()
