#!/usr/bin/env python
"""early_anchors step 6: filter the imagined rollouts, pick the dose, pack the Img and Real pair sets, build real + pairs.

  select  Filter: a branch is kept iff its imagined image coverage reaches filter.success_cov (0.9) at some round;
          its first such round is the hit. Candidate pairs of a kept branch: j = 0 .. min(hit, J_MAX), where frame j
          is NanoWM's frame after round j (the state after exec_acts[0 : FI*(j+1)]). Every round is used, re-plan or
          mid-chunk; the real anchor frame (the round-0 input) is never a pair. Sim success is NOT used (false
          positives stay). The dose is pack.dose x the real set's sampleable pairs, spread evenly over anchors
          (water-filling, nwaug.datagen.selection.water_fill); inside an anchor its kept branches are shuffled and
          visited round-robin, each visit drawing one unused j uniformly. One list for both arms: <out>/selection.json.
  pack    One LeRobotDataset per arm (`local/pusht_<tag>_img`, `local/pusht_<tag>_real`), one ep_len-9 episode per
          pair (train with drop_n_last_frames 8): row 0 image = imagined frame j (Img) or the simulator frame after
          replaying the same actions from the anchor state (Real), both 224 -> 96 via `pack_closed_loop.to96`;
          rows 0..8 = exec_acts[FI*(j+1) : FI*(j+1) + 9], clipped to the real action range. Same order in both arms.
          <repo>/meta/pair_index.json records branch, j, hit and the imagined / real image coverage of every pair.
  build   `<arm repo>_<mix>` = local/pusht_imgonly + the pairs, stats pinned to the real data. mix.mode balanced
          repeats the whole pair set so the pairs make ~mix.synth_frac of the samples; natural adds it once.

  uv run --extra training python research/nanowm_aug/proj/pusht/early_anchors/pack.py select \
      --rollout ~/datastorage/nanowm_pusht_aug/early_anchors/rollout_na10base/f2_n10
  uv run --extra training python .../pack.py pack  --rollout <same> --tag ea_f2n10
  uv run --extra training python .../pack.py build --tag ea_f2n10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ea_common import (
    CFG,
    EP_LEN,
    FI,
    add_pair,
    build_mix,
    create_pair_repo,
    first_hit,
    load_rollout,
    load_source,
    make_env,
    n_episodes,
    real_sampleable,
    replay_to_anchor,
)

from nwaug.attribution.config_io import load_config
from nwaug.datagen.pack_closed_loop import to96
from nwaug.datagen.selection import water_fill


# --------------------------------------------------------------------------------------------------------- select
def cmd_select(a) -> None:
    cfg = load_config(CFG)
    prov, d, _ = load_rollout(a.rollout, keys=("src_idx", "cand", "cov"))
    thr = cfg["filter"]["success_cov"]
    j_max = (prov["rounds"] * FI - EP_LEN) // FI - 1  # frame j needs FI*(j+1) + EP_LEN <= rounds*FI actions
    hit = first_hit(d["cov"], thr)
    kept = np.flatnonzero(hit >= 0)
    n = a.n or round(cfg["pack"]["dose"] * real_sampleable())
    per_anchor = defaultdict(list)  # anchor -> [(branch, [j ...])]
    for b in kept:
        per_anchor[int(d["src_idx"][b])].append((int(b), list(range(min(int(hit[b]), j_max) + 1))))
    rng = np.random.default_rng(cfg["pack"]["select_seed"])
    cap = {s: sum(len(js) for _, js in v) for s, v in per_anchor.items()}
    quota = water_fill(cap, n, rng)
    rows = []
    for s in sorted(per_anchor):
        order = [per_anchor[s][i] for i in rng.permutation(len(per_anchor[s]))]
        pools = [(b, list(rng.permutation(js))) for b, js in order]
        got, i = 0, 0
        while got < quota[s]:
            b, pool = pools[i % len(pools)]
            if pool:
                j = int(pool.pop())
                rows.append(
                    {
                        "anchor": s,
                        "branch": b,
                        "cand": int(d["cand"][b]),
                        "part": int(d["part"][b]),
                        "row": int(d["row"][b]),
                        "j": j,
                        "hit": int(hit[b]),
                        "img_cov": round(float(d["cov"][b, j]), 4),
                    }
                )
                got += 1
            i += 1
    out = a.rollout / "pack"
    out.mkdir(exist_ok=True)
    js = np.array([r["j"] for r in rows])
    meta = {
        "rollout": str(a.rollout),
        "n": len(rows),
        "seed": cfg["pack"]["select_seed"],
        "success_cov": thr,
        "j_max": j_max,
        "kept_branches": f"{len(kept)}/{len(hit)}",
        "anchors": len(per_anchor),
        "candidates": int(sum(cap.values())),
        "quota_hist": {
            str(k): int(v) for k, v in zip(*np.unique(list(quota.values()), return_counts=True), strict=True)
        },
        "j_hist": np.bincount(js, minlength=j_max + 1).tolist(),
        "img_cov_p10_50_90": np.percentile([r["img_cov"] for r in rows], [10, 50, 90]).round(3).tolist(),
        "pairs": rows,
    }
    (out / "selection.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "pairs"}, indent=1))


# ----------------------------------------------------------------------------------------------------------- pack
def real_frames(prov: dict, exec_acts: dict, sel: list[dict]) -> dict[tuple[int, int], np.ndarray]:
    """Simulator frame (224px) after FI*(j+1) executed actions from the anchor state, per selected (branch, j).
    Same replay as rollout.py fidelity: env.reset(env_seed) + a[0 : anchor_t], then the branch's exec_acts."""
    src = load_source(Path(prov["source"]))
    want = defaultdict(set)
    anchor = {}
    for r in sel:
        want[r["branch"]].add(r["j"])
        anchor[r["branch"]] = r["anchor"]
    env = make_env("pixels", 224, 400 + prov["rounds"] * FI)
    got = {}
    for n_done, (b, js) in enumerate(sorted(want.items())):
        replay_to_anchor(env, src, anchor[b])
        for s in range(FI * (max(js) + 1)):
            env.step(exec_acts[b][s])
            if (s + 1) % FI == 0 and (s + 1) // FI - 1 in js:
                got[(b, (s + 1) // FI - 1)] = np.asarray(env.unwrapped.get_obs())
        if (n_done + 1) % 200 == 0:
            print(f"  replayed {n_done + 1}/{len(want)} branches", flush=True)
    env.close()
    return got


def cmd_pack(a) -> None:
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from nwaug_envs.pusht.coverage import coverage_from_frame

    sel_meta = json.loads((a.rollout / "pack" / "selection.json").read_text())
    sel = sel_meta["pairs"]
    prov, _, parts = load_rollout(a.rollout, keys=("cov",))
    img, exec_acts = {}, {}
    by_part = defaultdict(list)
    for r in sel:
        by_part[r["part"]].append(r)
    for p, rs in by_part.items():
        z = np.load(parts[p])
        fr, ex = z["frames96"], z["exec_acts"]
        for r in rs:
            img[(r["branch"], r["j"])] = fr[r["row"], r["j"]]
            exec_acts[r["branch"]] = ex[r["row"]]
    real224 = real_frames(prov, exec_acts, sel)
    repos = {arm: f"local/pusht_{a.tag}_{arm}" for arm in ("img", "real")}
    ds = {arm: create_pair_repo(repo, a.overwrite) for arm, repo in repos.items()}
    index = []
    for e, r in enumerate(sel):
        b, j = r["branch"], r["j"]
        k0 = FI * (j + 1)
        f224 = real224[(b, j)]
        frames = {"img": img[(b, j)], "real": to96(f224)}
        for arm in ("img", "real"):
            add_pair(ds[arm], frames[arm], exec_acts[b][k0 : k0 + EP_LEN])
        real_cov = round(float(coverage_from_frame(f224)["coverage_cal"]), 4)
        index.append({"episode_index": e, **r, "k0": k0, "real_cov": real_cov})
        if (e + 1) % 200 == 0:
            print(f"  packed {e + 1}/{len(sel)}", flush=True)
    for arm, repo in repos.items():
        ds[arm].finalize()
        (HF_LEROBOT_HOME / repo / "meta" / "pair_index.json").write_text(
            json.dumps(
                {"arm": arm, "selection": str(a.rollout / "pack" / "selection.json"), "pairs": index},
                indent=1,
            )
        )
    gap = np.abs(np.array([r["img_cov"] - r["real_cov"] for r in index]))
    print(
        f"packed {len(sel)} pairs into {list(repos.values())}; |img - real| image cov at the pair frame: "
        f"mean {gap.mean():.3f}, p90 {np.percentile(gap, 90):.3f}"
    )


# ---------------------------------------------------------------------------------------------------------- build
def cmd_build(a) -> None:
    mix = load_config(CFG)["mix"]
    mode = a.mix or mix["mode"]
    for arm in ("img", "real"):
        pool = f"local/pusht_{a.tag}_{arm}"
        n_pool = n_episodes(pool)
        if mode == "balanced":  # pairs / (real + pairs) ~= synth_frac, one sample per pair episode
            rep = max(1, round(mix["synth_frac"] / (1 - mix["synth_frac"]) * real_sampleable() / n_pool))
        else:
            rep = 1
        repo = f"{pool}_{mode}"
        build_mix(pool, repo, rep)
        tot = n_episodes(repo)
        frac = n_pool * rep / (n_pool * rep + real_sampleable())
        print(f"{repo}: {tot} episodes ({n_pool} pairs x {rep} + real); pair share of samples {frac:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--rollout", type=Path, required=True, help="rollout.py run output dir")
    s.add_argument("--n", type=int, default=None, help="pairs (default: pack.dose x real sampleable pairs)")
    p = sub.add_parser("pack")
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--tag", default="ea_f2n10")
    p.add_argument("--overwrite", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--tag", default="ea_f2n10")
    b.add_argument("--mix", choices=["balanced", "natural"], default=None, help="default: mix.mode")
    a = ap.parse_args()
    {"select": cmd_select, "pack": cmd_pack, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
