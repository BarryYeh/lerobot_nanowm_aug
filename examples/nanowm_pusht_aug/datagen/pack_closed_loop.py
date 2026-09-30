#!/usr/bin/env python
"""Pack ACCEPTED closed-loop imagined rollouts into a LeRobotDataset (lerobot venv).

Successor to `pack_imagined.py`, which packs open-loop `imagined_rollout.py` shards at fps 2
with one action per frame. Closed-loop accepted rollouts have a different shape -- 8 imagined
frames 0.5 s apart plus 40 executed absolute targets at 0.1 s -- and must aggregate with the
fps-10 real dataset, so they need a different layout.

THE ALIGNMENT-SAFE fps-10 LAYOUT
--------------------------------
NanoWM only produces frames every `fi`(=5) sim steps, so a naive fps-10 pack would have to
zero-order-hold each imagined frame across 5 rows -- pairing a STALE image (up to 0.4 s old)
with a fresh action chunk on 4 of every 5 samples. That is avoidable:

  * emit each (imagined frame, action chunk) pair as its OWN `ep_len`-frame episode
  * train with `drop_n_last_frames = ep_len - 1`

`EpisodeAwareSampler` gives each episode valid sample starts `[from, to - drop_n_last)`, so an
episode of `drop+1` frames yields EXACTLY ONE sample, at its first frame. With `n_obs_steps=1`,
`observation_delta_indices == [0]`, so the later rows are never read as observations -- only
their `action` values are, to fill the horizon. Every emitted sample therefore pairs a FRESH
imagined frame with the actions that followed it. Zero staleness.

DEFAULT: ep_len=10 / drop=9 / pad-mode hold. Rows L0-L9 are written by this packer; L10-L15
(6 slots) are auto-padded by the dataset as a copy of L9. Hold-padding makes that copy the
agent's own position, so all 16 slots agree. drop=9 costs 0.9% of the real samples
(23,796 vs 24,002 at drop=8); ep_len=16 / drop=15 would remove auto-pad entirely but strips
the 1,442 highest-coverage real frames and measured 70.0% -> 59.5% on Control.
`drop` is global, so train BOTH the real-only and real+aug arms with the same value.

WHICH PAIRS ARE USABLE
----------------------
In `closed_loop_rollout.py` the policy sees the REAL anchor frame at round 0 and imagined frame
`j` at round `j+1`, so the executed chunk following imagined frame `j` starts at
`exec_abs_targets[fi*(j+1)]`. A full 16-action horizon needs `fi*(j+1) + 16 <= h_max*fi`,
i.e. `j <= h_max - 1 - 16/fi`: for h_max=8 that is j = 0..3, so **4 samples per rollout**.
The label is the ACTUALLY EXECUTED stream (a stitch of the re-planned chunks), which is what
produced the accepted outcome and therefore what rejection sampling should reinforce.

Actions are clipped to the real dataset's action range so an aggregated MIN_MAX normalizer
stays anchored to the real distribution.

PAIR INDEX SIDECAR
------------------
`<root>/meta/pair_index.json` maps each packed episode back to the branch it came from:
`{"accepted": ..., "pairs": [{"episode_index", "branch", "j", "k0", "cross", "n_real"}, ...]}`.
Data attribution (CUPID/TRAK) scores packed episodes, and this is the only record of which
branch / imagined round each one is. `episode_index` is local to THIS dataset: after
`aggregate_datasets(["local/pusht_imgonly", <this>])` add the real episode count (206).

`--index-only` writes the sidecar for an EXISTING pack without touching its data: rerun with
the exact original arguments (incl. `--sample-seed`); every episode's actions are checked
against the replayed selection, so mismatched arguments fail instead of mislabelling.

    uv run --extra training python examples/nanowm_pusht_aug/datagen/pack_closed_loop.py \
        --accepted examples/nanowm_pusht_aug/cl/accepted_t11 \
        --repo-id local/pusht_imgonly_cl_aug
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 -- the conventional torch alias

from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datagen import action_convert as _ac  # noqa: E402


def agent_track(anchor_state, tgts):
    """Agent xy before each executed action, via the same PD model the rollout used."""
    pos = np.asarray(anchor_state[:2], dtype=np.float64).copy()
    vel = np.zeros(2)
    out = np.empty((len(tgts) + 1, 2))
    for t in range(len(tgts)):
        out[t] = pos
        pos, vel = _ac.pd_step(pos, vel, tgts[t])
    out[len(tgts)] = pos
    return out


FEATURES = {
    "observation.image": {"dtype": "video", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"axes": ["x", "y"]},
    },  # names must match local/pusht_imgonly for aggregate_datasets
}
ACTION_LO = np.array([12.0, 25.0])  # local/pusht_imgonly action min
ACTION_HI = np.array([511.0, 511.0])  # local/pusht_imgonly action max


def to96(f224: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(f224).float().permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(96, 96), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def verify_existing_pack(root: Path, repo_id: str, pairs: list[dict], accepted: Path, fi: int) -> None:
    """Fail unless every episode of the pack at `root` starts with the actions `pairs` says it
    should. Only row 0 is checked: it is never padded and pins down (branch, j) uniquely."""
    hf = LeRobotDataset(repo_id, root=root).hf_dataset.select_columns(
        ["episode_index", "frame_index", "action"]
    )
    first = {
        int(e): np.asarray(a)
        for e, fr, a in zip(hf["episode_index"], hf["frame_index"], hf["action"], strict=True)
        if int(fr) == 0
    }
    if len(first) != len(pairs):
        raise SystemExit(f"pack has {len(first)} episodes but the replayed selection has {len(pairs)}")
    bad = [
        r["episode_index"]
        for r in pairs
        if not np.allclose(
            first[r["episode_index"]],
            np.clip(np.load(accepted / r["branch"])["exec_abs_targets"][r["k0"]], ACTION_LO, ACTION_HI),
            atol=1e-3,
        )
    ]
    if bad:
        raise SystemExit(
            f"{len(bad)} episodes do not match the replayed selection (first: {bad[:5]}); wrong args?"
        )
    print(f"verified: all {len(pairs)} episodes match the replayed selection")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accepted", type=Path, required=True)
    p.add_argument("--repo-id", default="local/pusht_imgonly_cl_aug")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--fi", type=int, default=5)
    p.add_argument("--horizon", type=int, default=16, help="policy horizon (actions the chunk spans)")
    p.add_argument(
        "--ep-len",
        type=int,
        default=10,
        help="frames written per imagined episode. Must equal drop_n_last_frames+1 so the "
        "sampler yields exactly ONE sample per episode. Default 10 pairs with "
        "drop_n_last_frames=9: 10 controllable slots, 6 auto-padded, -0.9%% real samples. "
        "(ep-len 9 + drop 8 keeps the real data identical to the baseline but auto-pads 7.) "
        "(ep-len 16 + drop 15 also gives one sample, but drop=15 strips the 1442 "
        "highest-coverage real observation frames -- mean coverage 0.608 vs 0.258 "
        "for the kept ones -- which damages the very regime this augmentation targets.)",
    )
    p.add_argument("--max-rollouts", type=int, default=None)
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Emit exactly this many pairs, uniformly subsampled (seeded) from all "
        "candidates. Use it to hold DOSE constant across arms so that a comparison "
        "isolates label quality instead of confounding it with data volume.",
    )
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument(
        "--frame-select",
        choices=["pre", "lastpush", "all"],
        default="pre",
        help="Which pre-crossing frames to keep (only with --truncate-cov). "
        "'pre' (default) needs >= ep-len genuine pre-crossing actions, i.e. "
        "cross-j >= 2 -- which SILENTLY DROPS j = cross-1, the frame right before "
        "the win. Measured: those excluded frames sit at median coverage 0.931 and "
        "account for 539 of the 1,013 frames in the [0.93,0.95) band where the "
        "policy visibly hesitates -- 53%% of the most relevant signal. "
        "'lastpush' keeps ONLY j = cross-1; 'all' keeps every pre-crossing frame. "
        "For those frames copy-pad is semantically right: the last pre-crossing "
        "target is the one that drives the block over the line, so holding it "
        "means commit-and-stay rather than hesitate.",
    )
    p.add_argument(
        "--pad-mode",
        choices=["copy", "zero", "hold"],
        default="hold",
        help="What to write into label slots with no genuine executed action (default hold). "
        "'copy' repeats the last real action (keeps pushing -- commands ~8 px of "
        "extra travel). 'zero' writes (0,0), which is an absolute TARGET at the "
        "top-left corner, i.e. a large out-of-range command, not a no-op. "
        "'hold' writes the agent's own position at that moment, so the PD "
        "controller is told to stay put -- the only value that actually means "
        "'stop'. With ep-len 10 the auto-pad of L10-L15 copies L9, which is "
        "already the hold value, so all 16 slots stay consistent.",
    )
    p.add_argument(
        "--frame-window",
        type=int,
        default=None,
        help="Keep frames j in [cross-K, cross] INCLUDING the crossing frame, and take "
        "the executed action stream as-is (no truncate-to-crossing, which measured "
        "+0.0 pp / p=1.00 and would make the j=cross label 100%% copy-pad since it "
        "has zero pre-crossing actions). Generalises 'lastpush': the window is a "
        "superset of j=cross-1, so it keeps that frame's benefit while restoring "
        "the run-up context that 'pre' discards and giving every pair 9 genuine "
        "actions instead of 5 real + 4 copies.",
    )
    p.add_argument(
        "--coherent-actions",
        type=int,
        default=None,
        help="Keep only the first N actions -- the ones that genuinely come from THIS "
        "frame's own policy inference -- and copy-pad the rest of the stored rows. "
        "The executed stream is a concatenation of fi-step blocks, each from a "
        "different inference, so a 9-action label straddles two: slots 1-5 are this "
        "frame's chunk, slots 6-9 belong to the NEXT frame's chunk (produced after "
        "the policy had already seen the next observation). At deployment all "
        "executed actions must come from ONE inference, so those later slots teach "
        "the policy to use information it will not have. Set 5 to isolate that.",
    )
    p.add_argument(
        "--truncate-cov",
        type=float,
        default=None,
        help="TRUNCATE-AT-CROSSING. Mirror the real env, which terminates the instant "
        "coverage > 0.95: cut each label at the first imagined round that reaches "
        "this value and keep ONLY pre-crossing actions. Without it ~51%% of every "
        "label is post-victory drift (measured), which teaches the policy not to "
        "commit to finishing. A pair is emitted only if it has >= ep-len genuine "
        "pre-crossing actions, so every stored row is real.",
    )
    p.add_argument(
        "--oracle-npz",
        type=Path,
        default=None,
        help="filter_roc npz with a `success` array aligned to sorted(accepted/*.npz). "
        "Keeps ONLY branches that verifiably cross 0.95 when their action stream is "
        "replayed in the real simulator -> 100%% label precision instead of 47.5%%.",
    )
    p.add_argument(
        "--pair-cov",
        type=float,
        default=0.95,
        help="keep pair j only if the 16 actions after imagined frame j demonstrably "
        "reach this coverage (max over the rounds they span). Without this, "
        "post-peak pairs teach actions that LOSE coverage.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--include-anchor",
        action="store_true",
        help="also pack (REAL anchor frame, exec_abs_targets[0:H]) per branch -- the pair a plain "
        "action-noise label would give without any world model. Needs branches written with "
        "anchor_frame (closed_loop_rollout.py). Sidecar marks it source=anchor, j=-1.",
    )
    p.add_argument(
        "--index-only",
        action="store_true",
        help="write meta/pair_index.json for an existing pack at --root/--repo-id without "
        "repacking. Pass the pack's original arguments; actions are verified per episode.",
    )
    args = p.parse_args()

    files = sorted(args.accepted.glob("*.npz"))
    if args.oracle_npz is not None:
        succ = np.load(args.oracle_npz)["success"]
        if len(succ) != len(files):
            raise SystemExit(f"oracle has {len(succ)} rows but {len(files)} branches -- must be the same run")
        files = [f for f, s in zip(files, succ, strict=True) if s]
        print(f"oracle filter: kept {len(files)} branches that really cross 0.95")
    if args.max_rollouts:
        files = files[: args.max_rollouts]
    if not files:
        raise SystemExit(f"no accepted rollouts in {args.accepted}")

    root = args.root or Path.home() / ".cache/huggingface/lerobot" / args.repo_id
    if args.index_only:
        if not root.exists():
            raise SystemExit(f"--index-only needs an existing pack at {root}")
        ds = None
    else:
        if root.exists():
            if not args.overwrite:
                raise SystemExit(f"{root} exists; pass --overwrite")
            shutil.rmtree(root)
        ds = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=FEATURES,
            root=root,
            robot_type="pusht_sim",
            use_videos=True,
        )

    H, fi = args.horizon, args.fi
    n_ep = n_clip = n_skip = 0
    EL = args.ep_len

    # ---- pass 1: enumerate every candidate pair (cheap; only metadata is touched) ----
    cands = []  # (file, j, k0, cross)
    for f in files:
        d = np.load(f)
        n_act = len(d["exec_abs_targets"])
        cov = d["coverage"]  # calibrated, per imagined round
        h_max = len(d["frames"])
        cross = None
        if args.truncate_cov is not None:
            c = np.nonzero(cov >= args.truncate_cov)[0]
            if len(c) == 0:
                n_skip += h_max
                continue  # never crosses -> nothing to learn from
            cross = int(c[0])
        if args.include_anchor:
            if "anchor_frame" not in d.files:
                raise SystemExit(f"{f.name} has no anchor_frame -- regenerate with closed_loop_rollout.py")
            cands.append((f, -1, 0, cross))  # the real anchor frame precedes the whole executed stream
        for j in range(h_max):
            k0 = fi * (j + 1)  # executed chunk following imagined frame j
            if cross is not None:
                if j > cross or (j == cross and args.frame_window is None):
                    break  # at/after the win -> no task left to demo
                if args.frame_window is not None:
                    if not (cross - args.frame_window <= j <= cross) or k0 + EL > n_act:
                        n_skip += 1
                        continue
                else:
                    gap = cross - j  # rounds between this frame and the win
                    keep = {"pre": gap >= 2, "lastpush": gap == 1, "all": gap >= 1}[args.frame_select]
                    if not keep:
                        n_skip += 1
                        continue
            else:
                if k0 + H > n_act:  # needs a FULL horizon, never pad
                    break
                span = cov[j + 1 : j + 1 + int(np.ceil(H / fi))]
                if len(span) == 0 or span.max() < args.pair_cov:
                    n_skip += 1
                    continue  # these actions do not reach success -> drop
            cands.append((f, j, k0, cross))
    print(f"candidate pairs: {len(cands)}")
    if args.max_pairs is not None:
        if len(cands) < args.max_pairs:
            raise SystemExit(f"only {len(cands)} candidates but --max-pairs {args.max_pairs} requested")
        idx = np.random.default_rng(args.sample_seed).choice(len(cands), args.max_pairs, replace=False)
        cands = [cands[i] for i in sorted(idx)]
        print(f"subsampled to {len(cands)} pairs (seed {args.sample_seed}) to hold dose constant")

    # ---- pass 2: write the selected pairs ----
    pairs = []  # sidecar rows, in packed-episode order
    for f, j, k0, cross in cands:
        d = np.load(f)
        tgts = d["exec_abs_targets"]
        n_act = len(tgts)
        n_real = H
        if cross is not None and args.frame_window is None:
            n_real = min(H, fi * (cross - j))  # genuine pre-crossing actions available
        n_real = min(n_real, max(0, n_act - k0))
        pairs.append(
            {
                "episode_index": len(pairs),
                "branch": f.name,
                "j": int(j),
                "k0": int(k0),
                "cross": None if cross is None else int(cross),
                "n_real": int(n_real),
                "source": "anchor" if j < 0 else "imagined",
                "inject_sigma": float(d["inject_sigma"]) if "inject_sigma" in d.files else 0.0,
            }
        )
        if args.index_only:
            continue
        track = agent_track(d["anchor_state"], tgts) if args.pad_mode == "hold" else None
        img = to96(d["anchor_frame"] if j < 0 else d["frames"][j])
        raw = tgts[k0 : k0 + H]
        if len(raw) < H:  # ran off the end of the executed stream
            raw = np.concatenate([raw, np.repeat(raw[-1:], H - len(raw), 0)], 0)
        chunk = np.clip(raw.copy(), ACTION_LO, ACTION_HI)
        if n_real < H:
            if args.pad_mode == "zero":
                chunk[n_real:] = 0.0
            elif args.pad_mode == "hold":
                chunk[n_real:] = np.clip(track[min(k0 + n_real, len(track) - 1)], ACTION_LO, ACTION_HI)
            else:
                chunk[n_real:] = chunk[max(n_real - 1, 0)]
        if args.coherent_actions is not None:
            n = args.coherent_actions
            chunk[n:] = chunk[n - 1]  # drop the later inference's actions; hold instead
        n_clip += int((raw[:n_real] != chunk[:n_real]).any(axis=1).sum())
        for u in range(EL):
            # only row 0's image is ever read as an observation (drop_n_last_frames = EL-1 and
            # n_obs_steps=1); the remaining rows exist solely to carry the action horizon
            ds.add_frame({"observation.image": img, "action": chunk[u].astype(np.float32), "task": "pusht"})
        ds.save_episode()
        n_ep += 1

    if args.index_only:
        verify_existing_pack(root, args.repo_id, pairs, args.accepted, fi)
    else:
        ds.finalize()
    (root / "meta" / "pair_index.json").write_text(
        json.dumps({"accepted": str(args.accepted), "pairs": pairs}, indent=1)
    )
    print(f"pair index: {len(pairs)} episodes -> {root / 'meta' / 'pair_index.json'}")
    if args.index_only:
        return

    print(
        f"packed {n_ep} episodes ({n_ep} training samples) from {len(files)} accepted rollouts "
        f"-> {args.repo_id}"
    )
    print(f"  {n_ep * EL} frames, episode length {EL}, fps {args.fps}, {n_clip} actions clipped to range")
    print(
        f"  dropped {n_skip} pairs (never crosses --truncate-cov, outside --frame-window, "
        f"excluded by --frame-select, or below --pair-cov {args.pair_cov})"
    )
    print(f"  root: {root}")
    print(f"\n  TRAIN WITH  --policy.drop_n_last_frames={EL - 1}  (both the real-only and real+aug arms)")


if __name__ == "__main__":
    main()
