#!/usr/bin/env python
"""SAILOR expand S7, arm B: the same branches with REAL simulator frames in place of the NanoWM frames.

For every branch npz in --src (the selected/ symlinks), replay it in a 224px gym_pusht env exactly as the labeler
does (place_seed -> lead_acts -> exec_abs_targets) and capture the frame after every round of FI steps, i.e. at
the step where the branch's imagined frame j was predicted (5(j+1)). The npz is written to --dst unchanged except
`frames` (real), `coverage` / `coverage_raw` (sim coverage) and a `frames_source="sim"` marker, so
pack_closed_loop.py packs both arms into the same (frame j, actions after j) pairs.

Asserted: the replayed anchor equals `anchor_state` and the agent pos/vel equals `agent_state_hist[k]` at every
round boundary within TOL (the label's checks). A branch that succeeds inside the replay keeps being stepped so
every round still has a frame (the actions after the success are part of the branch's action labels).
Diagnostic (written to <dst>/realize.json): block CoG distance between the imagined and the real frame per round,
from the same image estimator on both sides (coverage_from_frame.pose_img, scaled to 512 px).

    uv run --extra training python examples/nanowm_pusht_aug/critic/realize_branches.py --src <selected> --dst <realized>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent))
from datagen.coverage_from_frame import coverage_from_frame  # noqa: E402
from label_imagined_branches import FI, TOL, agent_pv, place_seed  # noqa: E402
from nwaug.pusht_state import read_body  # noqa: E402

IMG = 224


def realize(z: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Real frames (h_max,224,224,3) at steps FI*(k+1), sim coverage per frame, step of the first success (-1)."""
    h_max = len(z["frames"])
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels",
        observation_width=IMG,
        observation_height=IMG,
        max_episode_steps=10_000,
    )
    u = env.unwrapped
    place_seed(env, int(z["seed_index"]), z["seed_state"])
    for a in z["lead_acts"]:
        env.step(a)
    err = float(np.abs(read_body(u) - z["anchor_state"]).max())
    assert err <= TOL, f"replayed anchor differs from anchor_state by {err:.2e}"
    frames, cov, first = [], [], -1
    for t, a in enumerate(z["exec_abs_targets"][: h_max * FI]):
        _o, _r, term, _tr, info = env.step(a)
        if term and first < 0:
            first = t + 1
        if (t + 1) % FI == 0:
            k = (t + 1) // FI
            e = float(np.abs(agent_pv(u) - z["agent_state_hist"][k]).max())
            assert e <= TOL, f"agent state at round {k} differs from agent_state_hist by {e:.2e}"
            frames.append(np.asarray(u.get_obs(), np.uint8))
            cov.append(float(info["coverage"]))
    env.close()
    assert len(frames) == h_max, (len(frames), h_max)
    return np.stack(frames), np.asarray(cov, np.float32), first


def cog_px(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(coverage_from_frame(rgb)["pose_img"][:2], float) * (512.0 / IMG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="dir of branch npz (symlinks ok)")
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(a.src.glob("*.npz"))
    assert files, f"no npz under {a.src}"
    report = []
    for f in files:
        out = a.dst / f.name
        if out.exists():
            continue
        d = np.load(f)
        z = {k: d[k] for k in d.files}
        real, cov, first = realize(z)
        dist = [float(np.linalg.norm(cog_px(i) - cog_px(r))) for i, r in zip(z["frames"], real, strict=True)]
        z.update(frames=real, coverage=cov, coverage_raw=cov.copy(), frames_source=np.array("sim"))
        tmp = out.with_name(out.stem + ".partial.npz")
        np.savez_compressed(tmp, **z)
        tmp.rename(out)
        report.append({"branch": f.name, "success_in_replay_step": first, "cog_err_px_per_round": dist})
        print(
            f"{f.name}: success in replay @ {first}, imagined-vs-real block CoG err (px/512) "
            f"{', '.join(f'{x:.1f}' for x in dist)}",
            flush=True,
        )
    rp = a.dst / "realize.json"
    prev = json.loads(rp.read_text()) if rp.exists() else []
    allr = prev + report
    errs = np.array([e for r in allr for e in r["cog_err_px_per_round"]], float)
    summ = {
        "n": len(allr),
        "cog_err_median": float(np.nanmedian(errs)) if len(errs) else None,
        "cog_err_p90": float(np.nanpercentile(errs, 90)) if len(errs) else None,
        "branches": allr,
    }
    rp.write_text(json.dumps(summ, indent=1))
    print(
        f"{len(allr)} realized; imagined-vs-real CoG err median {summ['cog_err_median']} p90 {summ['cog_err_p90']} -> {a.dst}"
    )


if __name__ == "__main__":
    main()
