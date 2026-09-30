#!/usr/bin/env python
"""Step 5 of the value/reward-model plan: ensemble training on assembled stage-1 + stage-2 rows.

Plan: value-model-opus5-5-sonnet5-floofy-engelbart.md, sections "Model" and step 5. Do not change
label rules, splits or gate thresholds here -- report back instead if one looks wrong.

Loader: assembles one row table from every stage-1 shard (+its shard-aligned latents_*.npz) and
stage-2's labels.npz. Groups: stage-1 group = episode id; stage-2 group = seed_id (every branch,
round and sigma of one seed lands on the same side of the split).

Three-way grouped split, seeded by `train.split_seed`, done SEPARATELY within each source bucket
(band / reset / stage2) so each split preserves every bucket's proportion of the whole:
  val (20% of all groups)         held out for gates / val_predictions.npz ONLY -- never seen by any
                                   member's optimizer OR by its early-stopping model selection.
  es  (10% of the remaining 80%)  early-stopping set: model selection (which epoch's weights to keep)
                                   is done against this set, never against val. This is the fix for a
                                   leak Opus caught in the first version: early-stopping on val gave
                                   the gate metrics (and especially the shuffled-label control) an
                                   optimistic bias, because "best epoch" was being chosen by loss
                                   against the very labels the gate later scores against.
  fit (the rest)                  bootstrap-resampled per ensemble member and trained on.

Training rows: stage-2 wall_artifact rows are dropped from the loss (both the fit loss and the es
loss used for early stopping), but kept in val_predictions.npz (flagged) so value_gate_report.py can
still exclude them from gate metrics while leaving them inspectable. pre_success rows are NOT dropped
-- they train normally with y=1 (only gate metrics exclude them, per the label rules). Pose-aux loss
is masked to stage-1 rows only.

Ensemble: cfg model.ensemble members, each on a bootstrap resample of the FIT GROUPS (with
replacement, key = (source bucket, group id) so stage-1 episode ids and stage-2 seed ids never
collide), different seed per member, early stopping on the ES set (patience, per member, best weights
kept). Same procedure for the OracleNet baseline. One extra ValueNet member trains on labels shuffled
INDEPENDENTLY within the fit rows and within the es rows (so its own model selection never sees a
real label either) -- a leakage detector: its AUROC on val against the REAL labels should be ~0.5. A
second, independent sanity check verifies the AUROC code itself: the REAL (non-shuffled) value
ensemble's AUROC on val with the val labels permuted, which should also land at ~0.5.

    uv run --extra training python examples/nanowm_pusht_aug/critic/train_value_model.py \
        --config configs/value_model_topband.yaml --smoke
    uv run --extra training python examples/nanowm_pusht_aug/critic/train_value_model.py \
        --config configs/value_model_topband.yaml --smoke --overfit-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402
from value_model import (  # noqa: E402
    Normalizer,
    OracleEnsemble,
    OracleNet,
    ValueEnsemble,
    ValueNet,
    pose_target,
)

BUCKETS = ("band", "reset", "stage2")
BUCKET_OFFSET = {"band": 0, "reset": 1, "stage2": 2}
GROUP_MOD = 10_000_000  # >> any episode id (<=2000) or seed_id (<=250); keeps composite keys unique


# ============================================================================================ metrics
def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie-corrected ranks. NaN if either class is empty."""
    y_true = np.asarray(y_true, bool)
    score = np.asarray(score, np.float64)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    sorted_scores = score[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            sorted_ranks[i : j + 1] = sorted_ranks[i : j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks
    sum_ranks_pos = ranks[y_true].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def class_balance(y: np.ndarray, source: np.ndarray) -> dict:
    out = {"overall": {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else float("nan")}}
    for b in BUCKETS:
        m = source == b
        out[b] = {"n": int(m.sum()), "pos_rate": float(y[m].mean()) if m.any() else float("nan")}
    return out


# ============================================================================================ loading
def _load_stage1(stage1_dir: Path) -> dict:
    shard_paths = sorted(stage1_dir.glob("shard_*.npz"))
    assert shard_paths, f"no stage1 shards under {stage1_dir}"
    keys = (
        "latent_stack",
        "frame_steps",
        "agent_state",
        "cog_state",
        "cov",
        "y",
        "pre_success",
        "wall_artifact",
        "source",
        "episode",
        "seed_id",
        "r",
        "sigma",
        "stage",
    )
    cols = {k: [] for k in keys}
    for shard_path in shard_paths:
        idx = shard_path.stem.split("_")[-1]
        lat_path = shard_path.parent / f"latents_{idx}.npz"
        assert lat_path.exists(), f"missing {lat_path} -- run the encode stage before training"
        d = np.load(shard_path, allow_pickle=True)
        assert str(d["state_convention"]) == "cog", d["state_convention"]
        lat = np.load(lat_path)["latents"]  # (E,23,4,32,32) f16, shard-local rows

        episodes = d["episode"]
        ep_local = {int(e): i for i, e in enumerate(episodes)}
        wall_by_ep = dict(zip(episodes.tolist(), d["wall_artifact"].tolist(), strict=True))
        sigma_by_ep = dict(zip(episodes.tolist(), d["sigma"].tolist(), strict=True))

        q_ep, q_fs = d["q_episode"], d["q_frame_steps"]
        n_q = len(q_ep)
        assert not any(wall_by_ep[int(e)] for e in q_ep), (
            f"{shard_path}: a stage-1 query comes from a wall-artifact episode"
        )
        gsteps = q_fs // 5
        assert (gsteps * 5 == q_fs).all(), f"{shard_path}: q_frame_steps not on the 5-step grid"
        stack = np.empty((n_q, 3, 4, 32, 32), np.float16)
        for i in range(n_q):
            stack[i] = lat[ep_local[int(q_ep[i])], gsteps[i]]

        cols["latent_stack"].append(stack)
        cols["frame_steps"].append(q_fs.astype(np.int64))
        cols["agent_state"].append(d["q_agent_state"].astype(np.float64))
        cols["cog_state"].append(d["q_cog_state"].astype(np.float64))
        cols["cov"].append(d["q_coverage"].astype(np.float32))
        cols["y"].append(d["q_y"].astype(bool))
        cols["pre_success"].append(d["q_pre_success"].astype(bool))
        cols["wall_artifact"].append(np.zeros(n_q, bool))  # collection drops wall episodes entirely
        cols["source"].append(d["q_source"].astype("<U8"))
        cols["episode"].append(q_ep.astype(np.int64))
        cols["seed_id"].append(np.full(n_q, -1, np.int64))
        cols["r"].append(np.full(n_q, -1, np.int64))
        cols["sigma"].append(np.array([sigma_by_ep[int(e)] for e in q_ep], np.float32))
        cols["stage"].append(np.full(n_q, 1, np.int8))
    return {k: np.concatenate(v, axis=0) for k, v in cols.items()}


def _load_stage2(labels_path: Path) -> dict:
    assert labels_path.exists(), f"missing {labels_path} -- run the label stage before training"
    d = np.load(labels_path, allow_pickle=True)
    n = len(d["y"])
    return {
        "latent_stack": d["latent_stack"],
        "frame_steps": d["frame_steps"].astype(np.int64),
        "agent_state": d["agent_state"].astype(np.float64),
        "cog_state": d["replay_cog_state"].astype(np.float64),
        "cov": d["cov"].astype(np.float32),
        "y": d["y"].astype(bool),
        "pre_success": d["pre_success"].astype(bool),
        "wall_artifact": d["wall_artifact"].astype(bool),
        "source": np.full(n, "stage2", dtype="<U8"),
        "episode": np.full(n, -1, np.int64),
        "seed_id": d["seed_id"].astype(np.int64),
        "r": d["r"].astype(np.int64),
        "sigma": d["sigma"].astype(np.float32),
        "stage": np.full(n, 2, np.int8),
    }


def load_table(stage1_dir: Path, stage2_labels: Path) -> dict:
    s1, s2 = _load_stage1(stage1_dir), _load_stage2(stage2_labels)
    rows = {k: np.concatenate([s1[k], s2[k]], axis=0) for k in s1}
    fs = rows["frame_steps"]
    assert (np.diff(fs, axis=1) == 5).all(), "frame_steps spacing != 5 somewhere in the assembled table"
    for k in ("agent_state", "cog_state", "cov"):
        assert np.isfinite(rows[k]).all(), f"non-finite values in {k}"
    assert np.isfinite(rows["latent_stack"].astype(np.float32)).all(), "non-finite values in latent_stack"
    assert not (rows["wall_artifact"] & (rows["stage"] == 1)).any(), (
        "stage-1 rows must never be wall_artifact"
    )
    assert set(np.unique(rows["source"]).tolist()) <= set(BUCKETS), np.unique(rows["source"])
    return rows


# ============================================================================================ split
def compute_split(
    rows: dict, split_seed: int, val_frac: float, es_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Three-way grouped split, done separately per source bucket (see the module docstring):
    val_frac of all groups -> val (held out, gates only); es_frac of the remaining groups -> es
    (early stopping); the rest -> fit (bootstrap-resampled per member). Returns
    (val_mask, es_mask, fit_mask, group_key, info)."""
    group_key = np.where(rows["stage"] == 2, rows["seed_id"], rows["episode"]).astype(np.int64)
    val_mask = np.zeros(len(group_key), bool)
    es_mask = np.zeros(len(group_key), bool)
    fit_mask = np.zeros(len(group_key), bool)
    info = {}
    for bucket, off in BUCKET_OFFSET.items():
        bmask = rows["source"] == bucket
        if not bmask.any():
            info[bucket] = {
                "n_groups": 0,
                "n_val_groups": 0,
                "val_groups": [],
                "n_train_groups": 0,
                "n_es_groups": 0,
                "es_groups": [],
                "n_fit_groups": 0,
                "fit_groups": [],
            }
            continue
        groups = np.unique(group_key[bmask])
        rng = np.random.default_rng(split_seed * 1000 + off)
        perm = rng.permutation(len(groups))
        n_val = max(1, round(len(groups) * val_frac)) if len(groups) > 1 else 0
        val_groups, train_groups = groups[perm[:n_val]], groups[perm[n_val:]]

        # ES carve-out: a seed DERIVED FROM split_seed (offset +500, disjoint from the val rng's
        # +off stream) so it's deterministic and reproducible without colliding with the val split.
        rng_es = np.random.default_rng(split_seed * 1000 + off + 500)
        perm_es = rng_es.permutation(len(train_groups))
        n_es = max(1, round(len(train_groups) * es_frac)) if len(train_groups) > 1 else 0
        es_groups, fit_groups = train_groups[perm_es[:n_es]], train_groups[perm_es[n_es:]]

        info[bucket] = {
            "n_groups": int(len(groups)),
            "n_val_groups": int(len(val_groups)),
            "val_groups": [int(g) for g in val_groups],
            "n_train_groups": int(len(train_groups)),
            "n_es_groups": int(len(es_groups)),
            "es_groups": [int(g) for g in es_groups],
            "n_fit_groups": int(len(fit_groups)),
            "fit_groups": [int(g) for g in fit_groups],
        }
        val_mask |= bmask & np.isin(group_key, val_groups)
        es_mask |= bmask & np.isin(group_key, es_groups)
        fit_mask |= bmask & np.isin(group_key, fit_groups)
    return val_mask, es_mask, fit_mask, group_key, info


def composite_key(rows: dict, group_key: np.ndarray) -> np.ndarray:
    off = np.array([BUCKET_OFFSET[s] for s in rows["source"]], np.int64)
    return off * GROUP_MOD + group_key


def bootstrap_indices(train_idx: np.ndarray, ckey: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices for one ensemble member: bootstrap resample of the TRAIN groups (composite key),
    with replacement; a group drawn k times contributes its rows k times."""
    keys_here = ckey[train_idx]
    uniq = np.unique(keys_here)
    idx_by_key = {k: train_idx[keys_here == k] for k in uniq}
    draw = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([idx_by_key[k] for k in draw])


# ============================================================================================ model io
def forward_net(net, normalizer: Normalizer, rows: dict, idx: np.ndarray, device: str):
    agent_state = torch.as_tensor(rows["agent_state"][idx], dtype=torch.float32, device=device)
    if isinstance(net, ValueNet):
        latent = torch.as_tensor(rows["latent_stack"][idx].astype(np.float32), device=device)
        return net(normalizer.norm_latent(latent), normalizer.norm_agent_state(agent_state))
    cog_state = torch.as_tensor(rows["cog_state"][idx], dtype=torch.float32, device=device)
    return net(normalizer.oracle_feat(agent_state, cog_state))


def compute_loss(
    net,
    normalizer: Normalizer,
    rows: dict,
    pose_all: np.ndarray,
    idx: np.ndarray,
    device: str,
    w_reward: float,
    w_pose: float,
    y_override: np.ndarray | None = None,
):
    v_logit, rm_logit, pose_pred = forward_net(net, normalizer, rows, idx, device)
    y_np = rows["y"][idx] if y_override is None else y_override[idx]
    y = torch.as_tensor(y_np.astype(np.float32), device=device)
    cov = torch.as_tensor(rows["cov"][idx], dtype=torch.float32, device=device)
    bce = F.binary_cross_entropy_with_logits(v_logit, y)
    rm_mse = F.mse_loss(torch.sigmoid(rm_logit), cov)
    pose_mask = torch.as_tensor(rows["stage"][idx] == 1, device=device)
    if bool(pose_mask.any()):
        pose_tgt = torch.as_tensor(pose_all[idx], dtype=torch.float32, device=device)
        pose_mse = F.mse_loss(pose_pred[pose_mask], pose_tgt[pose_mask])
    else:
        pose_mse = torch.zeros((), device=device)
    loss = bce + w_reward * rm_mse + w_pose * pose_mse
    return loss, {
        "bce": float(bce.item()),
        "rm_mse": float(rm_mse.item()),
        "pose_mse": float(pose_mse.item()),
    }


def sample_batch(
    rng: np.random.Generator, stage1_idx: np.ndarray, stage2_idx: np.ndarray, batch: int, mix1: float
) -> np.ndarray:
    n1 = min(batch, max(1, round(batch * mix1))) if len(stage1_idx) else 0
    n2 = batch - n1
    parts = []
    if n1 and len(stage1_idx):
        parts.append(stage1_idx[rng.integers(0, len(stage1_idx), size=n1)])
    if n2 and len(stage2_idx):
        parts.append(stage2_idx[rng.integers(0, len(stage2_idx), size=n2)])
    if not parts:  # only one stage present at all -- fall back to whichever pool exists
        pool = stage1_idx if len(stage1_idx) else stage2_idx
        parts.append(pool[rng.integers(0, len(pool), size=batch)])
    return np.concatenate(parts)


def train_member(
    net,
    normalizer: Normalizer,
    rows: dict,
    pose_all: np.ndarray,
    fit_idx: np.ndarray,
    es_idx: np.ndarray,
    cfg_train: dict,
    device: str,
    seed: int,
    tag: str,
    y_override: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    """Trains on `fit_idx` (already the bootstrap-resampled pool for this member, if bootstrapping is
    on), early-stops on `es_idx` -- disjoint from `fit_idx` and from the held-out val split, so model
    selection never touches either the rows this member was fit on or the rows the gates score
    against."""
    stage1_fit = fit_idx[rows["stage"][fit_idx] == 1]
    s2_fit_all = fit_idx[rows["stage"][fit_idx] == 2]
    stage2_fit = s2_fit_all[~rows["wall_artifact"][s2_fit_all]]
    es_loss_idx = es_idx[~((rows["stage"][es_idx] == 2) & rows["wall_artifact"][es_idx])]
    assert len(es_loss_idx), f"{tag}: empty early-stopping set after dropping stage-2 wall_artifact rows"

    batch, mix1 = cfg_train["batch"], cfg_train["mix_stage1"]
    steps_per_epoch = max(1, int(len(stage1_fit) / max(1, round(batch * mix1))))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["wd"])
    rng = np.random.default_rng(seed)

    best_es, best_state, bad_epochs, history = float("inf"), None, 0, []
    for epoch in range(cfg_train["max_epochs"]):
        net.train()
        train_losses = []
        for _ in range(steps_per_epoch):
            idx = sample_batch(rng, stage1_fit, stage2_fit, batch, mix1)
            loss, _ = compute_loss(
                net,
                normalizer,
                rows,
                pose_all,
                idx,
                device,
                cfg_train["w_reward"],
                cfg_train["w_pose"],
                y_override,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        net.eval()
        with torch.no_grad():
            es_loss, _ = compute_loss(
                net,
                normalizer,
                rows,
                pose_all,
                es_loss_idx,
                device,
                cfg_train["w_reward"],
                cfg_train["w_pose"],
                y_override,
            )
        tl, el = float(np.mean(train_losses)), float(es_loss.item())
        if not (np.isfinite(tl) and np.isfinite(el)):
            raise RuntimeError(f"{tag}: non-finite loss at epoch {epoch} (train={tl}, es={el}) -- aborting")
        history.append({"epoch": epoch, "train_loss": tl, "es_loss": el})
        if el < best_es - 1e-6:
            best_es = el
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > cfg_train["patience"]:
                break
    net.load_state_dict(best_state)
    return history, best_es


@torch.no_grad()
def predict_v(net, normalizer: Normalizer, rows: dict, idx: np.ndarray, device: str) -> np.ndarray:
    net.eval()
    v_logit, _, _ = forward_net(net, normalizer, rows, idx, device)
    return torch.sigmoid(v_logit).cpu().numpy()


def train_ensemble(
    net_cls,
    net_kwargs: dict,
    rows: dict,
    pose_all: np.ndarray,
    normalizer: Normalizer,
    fit_idx: np.ndarray,
    es_idx: np.ndarray,
    val_idx: np.ndarray,
    ckey: np.ndarray,
    cfg_model: dict,
    cfg_train: dict,
    device: str,
    seed0: int,
    tag: str,
) -> tuple[list, list[list[dict]], list[float]]:
    """Bootstrap-resamples FIT groups per member (never the ES groups); early-stopping is always on
    the fixed `es_idx`; `val_idx` is used only to report each member's val AUROC (never for fitting or
    model selection)."""
    members, histories, val_aurocs = [], [], []
    for k in range(cfg_model["ensemble"]):
        seed = seed0 + k
        rng = np.random.default_rng(seed)
        idx_k = bootstrap_indices(fit_idx, ckey, rng) if cfg_model.get("bootstrap", True) else fit_idx
        net = net_cls(**net_kwargs).to(device)
        t0 = time.time()
        hist, best_es = train_member(
            net, normalizer, rows, pose_all, idx_k, es_idx, cfg_train, device, seed, f"{tag}[{k}]"
        )
        v_val = predict_v(net, normalizer, rows, val_idx, device)
        auc = roc_auc(rows["y"][val_idx], v_val)
        members.append(net)
        histories.append(hist)
        val_aurocs.append(auc)
        print(
            f"  {tag}[{k}] seed={seed} n_fit={len(idx_k)} epochs={len(hist)} "
            f"best_es_loss={best_es:.4f} val_auroc={auc:.4f} ({time.time() - t0:.1f}s)"
        )
    return members, histories, val_aurocs


# ============================================================================================ overfit test
def run_overfit_test(rows: dict, pose_all: np.ndarray, device: str, model_dir: Path) -> None:
    n = len(rows["y"])
    take = min(256, n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=take, replace=False)
    if take < 256:
        print(f"WARNING overfit-test: only {n} rows available (< 256); using all {take} of them")

    net = ValueNet().to(device)
    normalizer = Normalizer()
    normalizer.fit(rows["latent_stack"][idx], rows["agent_state"][idx])
    opt = torch.optim.AdamW(net.parameters(), lr=3.0e-4, weight_decay=0.0)

    bce_hist = []
    for _step in range(2000):
        loss, parts = compute_loss(net, normalizer, rows, pose_all, idx, device, w_reward=1.0, w_pose=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        bce_hist.append(parts["bce"])
        if parts["bce"] < 0.05:
            break
    final_bce = bce_hist[-1]
    passed = bool(final_bce < 0.05)
    print(
        f"overfit-test: {len(bce_hist)} steps on {take} rows, final train BCE={final_bce:.4f} -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "overfit_test.json").write_text(
        json.dumps({"n_rows": take, "steps": len(bce_hist), "final_bce": final_bce, "pass": passed}, indent=2)
    )


# ============================================================================================ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--overfit-test", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda requested but not available; falling back to cpu")
        device = "cpu"

    root = Path(cfg["out_dir"])
    stage1_dir = root / ("smoke/stage1" if args.smoke else "stage1")
    stage2_labels = root / ("smoke/stage2" if args.smoke else "stage2") / "labels.npz"
    model_dir = root / ("smoke/model" if args.smoke else "model")

    print(f"loading stage1 <- {stage1_dir}, stage2 <- {stage2_labels} (device={device})")
    t_load = time.time()
    rows = load_table(stage1_dir, stage2_labels)
    n1, n2 = int((rows["stage"] == 1).sum()), int((rows["stage"] == 2).sum())
    print(f"loaded {len(rows['y'])} rows (stage1={n1}, stage2={n2}) in {time.time() - t_load:.1f}s")
    pose_all = pose_target(rows["cog_state"])

    if args.overfit_test:
        run_overfit_test(rows, pose_all, device, model_dir)
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    ctr = cfg["train"]
    es_frac = ctr.get("es_frac", 0.1)
    val_mask, es_mask, fit_mask, group_key, split_info = compute_split(
        rows, ctr["split_seed"], ctr["val_frac"], es_frac
    )
    fit_idx, es_idx, val_idx = np.nonzero(fit_mask)[0], np.nonzero(es_mask)[0], np.nonzero(val_mask)[0]
    train_idx = np.nonzero(~val_mask)[0]  # fit + es combined -- normalizer stats only, never for gates
    ckey = composite_key(rows, group_key)
    (model_dir / "split.json").write_text(
        json.dumps(
            {
                "split_seed": ctr["split_seed"],
                "val_frac": ctr["val_frac"],
                "es_frac": es_frac,
                "n_train_rows": int(len(train_idx)),
                "n_fit_rows": int(len(fit_idx)),
                "n_es_rows": int(len(es_idx)),
                "n_val_rows": int(len(val_idx)),
                "buckets": split_info,
            },
            indent=2,
        )
    )
    print(f"split: {len(fit_idx)} fit rows / {len(es_idx)} es rows / {len(val_idx)} val rows")
    for b, info in split_info.items():
        print(
            f"  {b}: {info['n_groups']} groups ({info['n_val_groups']} val, "
            f"{info['n_es_groups']} es, {info['n_fit_groups']} fit)"
        )

    normalizer = Normalizer()
    normalizer.fit(rows["latent_stack"][train_idx], rows["agent_state"][train_idx])

    cfg_model = cfg["model"]
    value_kwargs = {
        "conv": tuple(cfg_model["conv"]),
        "feat": cfg_model["feat"],
        "hidden": cfg_model["hidden"],
    }
    oracle_kwargs = {"hidden": cfg_model["hidden"]}

    print(f"\n=== training ValueNet ensemble (K={cfg_model['ensemble']}) ===")
    value_members, value_hist, value_aurocs = train_ensemble(
        ValueNet,
        value_kwargs,
        rows,
        pose_all,
        normalizer,
        fit_idx,
        es_idx,
        val_idx,
        ckey,
        cfg_model,
        ctr,
        device,
        1000,
        "value",
    )
    print(f"\n=== training OracleNet ensemble (K={cfg_model['ensemble']}) ===")
    oracle_members, oracle_hist, oracle_aurocs = train_ensemble(
        OracleNet,
        oracle_kwargs,
        rows,
        pose_all,
        normalizer,
        fit_idx,
        es_idx,
        val_idx,
        ckey,
        cfg_model,
        ctr,
        device,
        2000,
        "oracle",
    )

    print("\n=== shuffled-label control (1 ValueNet member, fit AND es labels shuffled independently) ===")
    # Shuffle fit-pool and es-pool labels with SEPARATE rngs so model selection (early stopping on es)
    # never sees a real label either -- the leak Opus flagged in the first version. val is untouched.
    y_shuffled = rows["y"].copy()
    shuf_fit_pool = fit_idx[~((rows["stage"][fit_idx] == 2) & rows["wall_artifact"][fit_idx])]
    shuf_es_pool = es_idx[~((rows["stage"][es_idx] == 2) & rows["wall_artifact"][es_idx])]
    y_shuffled[shuf_fit_pool] = np.random.default_rng(3000).permutation(y_shuffled[shuf_fit_pool])
    y_shuffled[shuf_es_pool] = np.random.default_rng(3001).permutation(y_shuffled[shuf_es_pool])
    shuffled_net = ValueNet(**value_kwargs).to(device)
    shuffled_hist, shuffled_best_es = train_member(
        shuffled_net,
        normalizer,
        rows,
        pose_all,
        fit_idx,
        es_idx,
        ctr,
        device,
        3000,
        "shuffled",
        y_shuffled,
    )
    shuffled_val_v = predict_v(shuffled_net, normalizer, rows, val_idx, device)
    shuffled_auroc = roc_auc(rows["y"][val_idx], shuffled_val_v)  # vs REAL y: leakage check, expect ~0.5
    print(
        f"  shuffled[0] best_es_loss={shuffled_best_es:.4f} val_auroc_vs_real_y={shuffled_auroc:.4f} (expect ~0.5)"
    )

    ve = ValueEnsemble(value_members, normalizer)
    oe = OracleEnsemble(oracle_members, normalizer)
    ve.save(model_dir / "value_ensemble.pt")
    oe.save(model_dir / "oracle_ensemble.pt")
    print(f"\nsaved {model_dir / 'value_ensemble.pt'}, {model_dir / 'oracle_ensemble.pt'}")

    # ---- val predictions ----
    ve.eval()
    oe.eval()
    with torch.no_grad():
        latent = torch.as_tensor(rows["latent_stack"][val_idx].astype(np.float32), device=device)
        agent_state = torch.as_tensor(rows["agent_state"][val_idx], dtype=torch.float32, device=device)
        cog_state = torch.as_tensor(rows["cog_state"][val_idx], dtype=torch.float32, device=device)
        v_logit, rm_logit, pose_pred = ve._forward_all(latent, agent_state)
        v_sig, rm_sig = torch.sigmoid(v_logit), torch.sigmoid(rm_logit)
        ov_logit, _, _ = oe._forward_all(agent_state, cog_state)
        oracle_v_mean_all = torch.sigmoid(ov_logit).mean(0).cpu().numpy()

    val_pred = {
        "v_per_member": v_sig.cpu().numpy().T,
        "v_mean_all": v_sig.mean(0).cpu().numpy(),
        "v_std": v_sig.std(0).cpu().numpy(),
        "rm_mean_all": rm_sig.mean(0).cpu().numpy(),
        "rm_std": rm_sig.std(0).cpu().numpy(),
        "pose_pred": pose_pred.mean(0).cpu().numpy(),
        "oracle_v_mean_all": oracle_v_mean_all,
        "y": rows["y"][val_idx],
        "cov": rows["cov"][val_idx],
        "pre_success": rows["pre_success"][val_idx],
        "wall_artifact": rows["wall_artifact"][val_idx],
        "source": rows["source"][val_idx],
        "stage": rows["stage"][val_idx],
        "r": rows["r"][val_idx],
        "sigma": rows["sigma"][val_idx],
        "seed_id": rows["seed_id"][val_idx],
        "episode": rows["episode"][val_idx],
        "cog_state": rows["cog_state"][val_idx],
    }
    np.savez_compressed(model_dir / "val_predictions.npz", **val_pred)
    print(f"wrote {model_dir / 'val_predictions.npz'} ({len(val_idx)} rows)")

    # ---- metric-code sanity check: the REAL ensemble's AUROC against PERMUTED val labels. This
    # doesn't touch training at all -- it only checks that roc_auc() itself isn't silently biased
    # high, independent of the shuffled-label-control member above. Expect ~0.5. ----
    permuted_val_y = np.random.default_rng(4000).permutation(rows["y"][val_idx])
    metric_check_auroc = roc_auc(permuted_val_y, val_pred["v_mean_all"])
    print(
        f"metric-code sanity check: real ensemble AUROC vs PERMUTED val labels = {metric_check_auroc:.4f} (expect ~0.5)"
    )

    # ---- health checks / train_log.json ----
    def member_gap(hist: list[dict]) -> float:
        best = min(hist, key=lambda h: h["es_loss"])
        return float(best["es_loss"] - best["train_loss"])

    train_log = {
        "n_rows": {"total": len(rows["y"]), "stage1": n1, "stage2": n2},
        "split": {
            "n_fit": int(len(fit_idx)),
            "n_es": int(len(es_idx)),
            "n_val": int(len(val_idx)),
        },
        "class_balance": {
            "fit": class_balance(rows["y"][fit_idx], rows["source"][fit_idx]),
            "es": class_balance(rows["y"][es_idx], rows["source"][es_idx]),
            "val": class_balance(rows["y"][val_idx], rows["source"][val_idx]),
        },
        "value_ensemble": {
            "per_member_val_auroc": value_aurocs,
            "per_member_final_gap": [member_gap(h) for h in value_hist],
            "history": value_hist,
        },
        "oracle_ensemble": {
            "per_member_val_auroc": oracle_aurocs,
            "per_member_final_gap": [member_gap(h) for h in oracle_hist],
            "history": oracle_hist,
        },
        "shuffled_label_control": {
            "val_auroc_vs_real_y": shuffled_auroc,
            "expected": "~0.5 (leakage detector: fit AND es labels were shuffled independently, so "
            "this member's training AND its early-stopping model selection never saw a real label)",
            "final_gap": member_gap(shuffled_hist),
            "history": shuffled_hist,
        },
        "metric_code_sanity_check": {
            "desc": "the REAL (non-shuffled) value ensemble's AUROC on val with the val LABELS "
            "permuted -- checks roc_auc() itself, independent of the shuffled-label-control member",
            "auroc_vs_permuted_val_labels": metric_check_auroc,
            "expected": "~0.5",
        },
        "nan_guard": "train_member raises RuntimeError and aborts the run on any non-finite train/es loss",
    }
    (model_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))
    print(f"wrote {model_dir / 'train_log.json'}")

    # ---- curves.png ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, hist_list, title in (
        (axes[0], value_hist, "ValueNet members"),
        (axes[1], oracle_hist, "OracleNet members"),
        (axes[2], [shuffled_hist], "shuffled-label control"),
    ):
        for k, hist in enumerate(hist_list):
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["train_loss"] for h in hist], color=f"C{k}", linestyle="--", alpha=0.7)
            ax.plot(epochs, [h["es_loss"] for h in hist], color=f"C{k}", linestyle="-", label=f"member {k}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (-- train, - early-stopping set)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(model_dir / "curves.png", dpi=120)
    plt.close(fig)
    print(f"wrote {model_dir / 'curves.png'}")

    print(
        f"\nvalue AUROC per member: {[round(a, 3) for a in value_aurocs]}\n"
        f"oracle AUROC per member: {[round(a, 3) for a in oracle_aurocs]}\n"
        f"shuffled-label control AUROC vs real y: {shuffled_auroc:.3f} (expect ~0.5)\n"
        f"metric-code sanity check (real ensemble vs permuted val y): {metric_check_auroc:.3f} (expect ~0.5)"
    )


if __name__ == "__main__":
    main()
