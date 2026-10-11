#!/usr/bin/env python
"""Last-mile experiment, step 3: choose one candidate chunk per stuck start from IMAGINED frames only.

Candidates = every branch under <root>/sigma*/branches (closed_loop_rollout, h_max 2: one 10-step chunk sampled on the
real anchor frame, executed with action noise sigma). Score = max imagined coverage over the branch's rounds (the
NanoWM worker's calibrated `coverage`). A start is accepted when some candidate scores >= --thr and the real anchor
frame's own estimate is < --thr (not already solved). Per accepted start:
  W  the highest-scoring candidate (ties -> lowest sigma, then cand)
  R  a uniformly random candidate of the same start (rng seed --seed)
Writes flat symlink dirs <out>/W and <out>/R (<sigma dir>_<branch file>, as pack_closed_loop expects) and
<out>/selection.json. Exit code 1 if fewer than --min-accepted starts are accepted (gate 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC.parent))
from datagen.coverage_from_frame import coverage_from_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="dir holding sigma*/branches")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=960000)
    ap.add_argument("--min-accepted", type=int, default=100)
    ap.add_argument(
        "--force-min",
        type=int,
        default=0,
        help="SMOKE ONLY: if fewer starts are accepted, force the best-scoring ones (flagged forced=true)",
    )
    a = ap.parse_args()
    by_start: dict[int, list[dict]] = {}
    anchor_est: dict[int, float] = {}
    for f in sorted(a.root.glob("sigma*/branches/seed*_cand*.npz")):
        z = np.load(f)
        sid = int(z["seed_index"])
        rec = {
            "path": f,
            "name": f"{f.parent.parent.name}_{f.name}",
            "sigma": float(z["inject_sigma"]),
            "cand": int(f.stem.split("_cand")[1]),
            "score": float(np.max(z["coverage"])),
        }
        if sid not in anchor_est:  # the real anchor frame is identical across a start's candidates
            anchor_est[sid] = float(coverage_from_frame(z["anchor_frame"])["coverage_cal"])
        by_start.setdefault(sid, []).append(rec)
    starts = sorted(by_start)
    rng = np.random.default_rng(a.seed)
    rows = []
    for s in starts:
        cands = sorted(by_start[s], key=lambda r: (r["sigma"], r["cand"]))
        a_est = anchor_est[s]
        best = max(
            cands, key=lambda r: r["score"]
        )  # max() keeps the first of equal scores -> lowest sigma, cand
        accepted = best["score"] >= a.thr and a_est < a.thr
        rnd = cands[
            int(rng.integers(len(cands)))
        ]  # drawn for every start so R does not depend on acceptance order
        rows.append(
            {
                "start": s,
                "n_cands": len(cands),
                "anchor_est": a_est,
                "accepted": bool(accepted),
                "W": best["name"],
                "W_score": best["score"],
                "W_sigma": best["sigma"],
                "R": rnd["name"],
                "R_score": rnd["score"],
                "R_sigma": rnd["sigma"],
                "n_cands_ge_thr": int(sum(r["score"] >= a.thr for r in cands)),
                "_paths": {"W": str(best["path"]), "R": str(rnd["path"])},
            }
        )
    acc = [r for r in rows if r["accepted"]]
    if len(acc) < a.force_min:  # plumbing test only
        for r in sorted((r for r in rows if not r["accepted"]), key=lambda r: -r["W_score"])[
            : a.force_min - len(acc)
        ]:
            r["accepted"], r["forced"] = True, True
        acc = [r for r in rows if r["accepted"]]
    for arm in ("W", "R"):
        d = a.out / arm
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
        for r in acc:
            (d / r[arm]).symlink_to(Path(r["_paths"][arm]).resolve())
    summ = {
        "n_starts": len(rows),
        "n_accepted": len(acc),
        "thr": a.thr,
        "rejected_anchor_already_solved": int(sum(r["anchor_est"] >= a.thr for r in rows)),
        "W_sigma": {
            str(s): int(sum(r["W_sigma"] == s for r in acc)) for s in sorted({r["W_sigma"] for r in acc})
        },
        "mean_cands_ge_thr_accepted": float(np.mean([r["n_cands_ge_thr"] for r in acc])) if acc else 0.0,
        "rows": [{k: v for k, v in r.items() if k != "_paths"} for r in rows],
    }
    (a.out / "selection.json").write_text(json.dumps(summ, indent=1))
    print(
        f"accepted {len(acc)}/{len(rows)} starts (anchor already >= thr: {summ['rejected_anchor_already_solved']}); "
        f"W sigma {summ['W_sigma']}; -> {a.out}"
    )
    if len(acc) < a.min_accepted:
        print(f"GATE 1 FAILED: {len(acc)} < {a.min_accepted} accepted starts")
        sys.exit(1)


if __name__ == "__main__":
    main()
