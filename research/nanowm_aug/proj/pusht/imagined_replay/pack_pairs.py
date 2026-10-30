#!/usr/bin/env python
"""imagined_replay step 4a: pick the dose, pack the Real and Imagined pair sets, build real + pairs.

select  Spread `--n` pairs evenly over the starts that have trajectories passing the fidelity filter
        (water-filling: a start with too few pairs gives all it has, the rest is shared by the others; the
        remainder goes +1 to randomly chosen starts). Inside a start, its passing trajectories are shuffled and
        visited round-robin, each visit drawing one unused s at random. One list for both arms:
        <out>/selection.json.
pack    One LeRobotDataset per arm (`local/pusht_<tag>_real`, `local/pusht_<tag>_img`) in the last-mile / method1
        layout: one ep_len-9 episode per pair (train with drop_n_last_frames 8), row 0 image = F(s) or F̂(s)
        (224 -> 96, `pack_closed_loop.to96`), rows 0..8 = recorded a[s : s+9], rows past T copy a[T-1] (the real
        dataset's episode-end behaviour), clipped to the real action range. Same episode order in both arms.
build   `<arm repo>_plus_real` = local/pusht_imgonly + the pairs, stats pinned to the real data (method1's build).

  uv run --extra training python .../pack_pairs.py select --root ~/datastorage/nanowm_pusht_aug/imagined_replay --n 1263
  uv run --extra training python .../pack_pairs.py pack   --root ... --tag ir
  uv run --extra training python .../pack_pairs.py build  --tag ir
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from ir_common import EP_LEN, FI, REAL_REPO, SELECT_SEED

from nwaug.datagen.selection import water_fill


def cmd_select(a) -> None:
    passing = json.loads((a.root / "imagine" / "fidelity_filter.json").read_text())["pass"]
    trajs = defaultdict(list)  # start -> [(name, [s ...])]
    for name in sorted(n for n, ok in passing.items() if ok):
        z = np.load(a.root / "collect" / "traj" / name)
        ss = list(range(int(z["t_start"]) + 1, int(z["T"]) - FI + 1))
        trajs[int(re.match(r"start(\d+)_", name).group(1))].append((name, ss))
    rng = np.random.default_rng(SELECT_SEED)
    cap = {s: sum(len(ss) for _, ss in v) for s, v in trajs.items()}
    quota = water_fill(cap, a.n, rng)
    rows = []
    for st in sorted(trajs):
        order = [trajs[st][i] for i in rng.permutation(len(trajs[st]))]
        pools = [(name, list(rng.permutation(ss))) for name, ss in order]
        got, i = 0, 0
        while got < quota[st]:
            name, pool = pools[i % len(pools)]
            if pool:
                rows.append({"start_id": st, "traj": name, "s": int(pool.pop())})
                got += 1
            i += 1
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "n": len(rows),
        "seed": SELECT_SEED,
        "starts": len(trajs),
        "quota_hist": {
            str(k): int(v) for k, v in zip(*np.unique(list(quota.values()), return_counts=True), strict=True)
        },
        "pairs": rows,
    }
    (a.out / "selection.json").write_text(json.dumps(meta, indent=1))
    print(
        f"selected {len(rows)} pairs over {len(trajs)} starts; quota per start -> count {meta['quota_hist']}"
    )


def imagined_frames(imagine_all: Path, want: set[tuple[str, int]]) -> dict[tuple[str, int], np.ndarray]:
    got = {}
    for p in sorted(imagine_all.glob("part*_c*.npz")):
        z = np.load(p)
        idx = [i for i, k in enumerate(zip(z["traj"].tolist(), z["s"].tolist(), strict=True)) if k in want]
        if idx:
            fr = z["frames_img"]
            for i in idx:
                got[(str(z["traj"][i]), int(z["s"][i]))] = fr[i]
    missing = want - set(got)
    if missing:
        raise SystemExit(
            f"{len(missing)} selected windows have no imagined frame, e.g. {sorted(missing)[:3]}"
        )
    return got


def label(z: dict, s: int) -> tuple[np.ndarray, int]:
    """Recorded a[s : s+9]; rows past T copy a[T-1], as the real dataset does at an episode end."""
    a0, acts = int(z["action_steps"][0]), z["actions"]
    raw = acts[s - a0 : s - a0 + EP_LEN]
    n_real = len(raw)
    if n_real < EP_LEN:
        raw = np.concatenate([raw, np.repeat(acts[-1:], EP_LEN - n_real, 0)], 0)
    return raw, n_real


def cmd_pack(a) -> None:
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from nwaug.datagen.pack_closed_loop import add_pair, create_pair_repo, to96

    sel = json.loads((a.root / "pack" / "selection.json").read_text())["pairs"]
    img = imagined_frames(a.root / "imagine" / "all", {(r["traj"], r["s"]) for r in sel})
    repos = {arm: f"local/pusht_{a.tag}_{arm}" for arm in ("real", "img")}
    ds = {arm: create_pair_repo(repo, a.overwrite) for arm, repo in repos.items()}
    index, cache = [], {}
    for e, r in enumerate(sel):
        if r["traj"] not in cache:
            cache.clear()
            cache[r["traj"]] = dict(np.load(a.root / "collect" / "traj" / r["traj"]))
        z = cache[r["traj"]]
        s, f0 = r["s"], int(z["frame_steps"][0])
        chunk, n_real = label(z, s)
        frames = {"real": to96(z["frames224"][s - f0]), "img": to96(img[(r["traj"], s)])}
        for arm in ("real", "img"):
            add_pair(ds[arm], frames[arm], chunk)
        index.append({"episode_index": e, **r, "T": int(z["T"]), "n_real": n_real})
        if (e + 1) % 200 == 0:
            print(f"  packed {e + 1}/{len(sel)}", flush=True)
    for arm, repo in repos.items():
        ds[arm].finalize()
        (HF_LEROBOT_HOME / repo / "meta" / "pair_index.json").write_text(
            json.dumps(
                {"arm": arm, "selection": str(a.root / "pack" / "selection.json"), "pairs": index}, indent=1
            )
        )
    print(
        f"packed {len(sel)} pairs into {list(repos.values())}; labels with copy-pad rows: "
        f"{sum(r['n_real'] < EP_LEN for r in index)}"
    )


def cmd_build(a) -> None:
    from nwaug.attribution.build_selected_datasets import build_all_plus_real

    for arm in ("real", "img"):
        pool = f"local/pusht_{a.tag}_{arm}"
        n_pool, n = build_all_plus_real(REAL_REPO, pool)
        print(f"{pool}_plus_real: {n} episodes ({n_pool} pairs + real)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--n", type=int, required=True)
    p = sub.add_parser("pack")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tag", default="ir")
    p.add_argument("--overwrite", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--tag", default="ir")
    a = ap.parse_args()
    if a.cmd == "select":
        a.out = a.root / "pack"
    {"select": cmd_select, "pack": cmd_pack, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
