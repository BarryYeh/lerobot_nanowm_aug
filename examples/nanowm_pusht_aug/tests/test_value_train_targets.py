"""SAILOR v1 T5: V head = BCE with soft labels on v_target; empty band bucket; loaders keep the new fields."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402, N812
from train_value_model import (  # noqa: E402
    BUCKETS,
    class_balance,
    compute_loss,
    compute_split,
    load_stage2_rows,
    predict_rows,
)
from value_model import (  # noqa: E402
    Normalizer,
    OracleEnsemble,
    OracleNet,
    ValueEnsemble,
    ValueNet,
    pose_target,
)


def _rows(n=64, seed=0, sources=("reset", "stage2")):
    rng = np.random.default_rng(seed)
    stage = np.where(np.arange(n) % 2 == 0, 1, 2).astype(np.int8)
    y = rng.random(n) < 0.5
    return {
        "agent_state": rng.uniform(50, 450, (n, 4)),
        "cog_state": np.concatenate([rng.uniform(50, 450, (n, 4)), rng.uniform(-3, 3, (n, 1))], axis=1),
        "latent_stack": rng.standard_normal((n, 3, 4, 32, 32)).astype(np.float16),
        "y": y,
        "v_target": np.where(y, 0.995 ** rng.integers(1, 200, n), 0.0).astype(np.float32),
        "cov": rng.random(n).astype(np.float32),
        "stage": stage,
        "wall_artifact": np.zeros(n, bool),
        "source": np.where(stage == 1, sources[0], sources[1]).astype("<U8"),
        "episode": np.where(stage == 1, np.arange(n) // 4, -1).astype(np.int64),
        "seed_id": np.where(stage == 2, np.arange(n) // 4, -1).astype(np.int64),
    }


def test_v_loss_is_soft_label_bce_against_v_target_and_rm_regresses_cov():
    rows = _rows()
    pose_all = pose_target(rows["cog_state"])
    net = OracleNet()
    norm = Normalizer()
    idx = np.arange(len(rows["y"]))
    loss, parts = compute_loss(net, norm, rows, pose_all, idx, "cpu", w_reward=1.0, w_pose=0.5)
    feat = norm.oracle_feat(
        torch.as_tensor(rows["agent_state"], dtype=torch.float32),
        torch.as_tensor(rows["cog_state"], dtype=torch.float32),
    )
    v_logit, rm_logit, _ = net(feat)
    soft = F.binary_cross_entropy_with_logits(v_logit, torch.as_tensor(rows["v_target"]))
    hard = F.binary_cross_entropy_with_logits(v_logit, torch.as_tensor(rows["y"].astype(np.float32)))
    assert parts["bce"] == pytest.approx(float(soft), rel=1e-5)
    assert abs(parts["bce"] - float(hard)) > 1e-3  # the target really is the soft one
    assert parts["rm_mse"] == pytest.approx(
        float(F.mse_loss(torch.sigmoid(rm_logit), torch.as_tensor(rows["cov"]))), rel=1e-5
    )
    # an override replaces the whole V target (shuffled control / binary overfit test)
    _, p2 = compute_loss(
        net, norm, rows, pose_all, idx, "cpu", 1.0, 0.5, y_override=rows["y"].astype(np.float32)
    )
    assert p2["bce"] == pytest.approx(float(hard), rel=1e-5)
    # the loss is minimised by predicting v_target itself (soft-label BCE has its optimum at p = target)
    t = torch.as_tensor(rows["v_target"]).clamp(1e-4, 1 - 1e-4)
    at_target = F.binary_cross_entropy_with_logits(torch.logit(t), torch.as_tensor(rows["v_target"]))
    assert float(at_target) <= float(soft) + 1e-6


def test_empty_band_bucket_does_not_crash_split_or_balance():
    assert "band" in BUCKETS
    rows = _rows(n=80)  # sources: reset + stage2 only -> band is empty
    val, es, fit, _, info = compute_split(rows, split_seed=0, val_frac=0.2)
    assert info["band"]["n_groups"] == 0 and info["band"]["n_val_groups"] == 0
    assert (val | es | fit).all() and not (val & fit).any() and not (es & fit).any()
    cb = class_balance(rows["y"], rows["source"])
    assert cb["band"]["n"] == 0 and np.isnan(cb["band"]["pos_rate"]) and cb["reset"]["n"] > 0


def test_predict_rows_and_stage2_loader_carry_new_fields(tmp_path):
    n = 12
    rng = np.random.default_rng(0)
    y = rng.random(n) < 0.5
    np.savez(
        tmp_path / "labels.npz",
        latent_stack=rng.standard_normal((n, 3, 4, 32, 32)).astype(np.float16),
        frame_steps=np.tile(np.array([0, 5, 10]), (n, 1)),
        agent_state=rng.uniform(50, 450, (n, 4)),
        cov=rng.random(n).astype(np.float32),
        y=y,
        pre_success=np.zeros(n, bool),
        wall_artifact=np.zeros(n, bool),
        sigma=np.zeros(n, np.float32),
        seed_id=np.arange(n) // 4,
        r=np.zeros(n, np.int64),
        replay_cog_state=np.concatenate([rng.uniform(50, 450, (n, 4)), rng.uniform(-3, 3, (n, 1))], axis=1),
        v_target=(y * 0.5).astype(np.float32),
        T=np.where(y, 100, -1),
        success_step=np.where(y, 100, -1),
        cov_end=rng.random(n).astype(np.float32),
        cand=np.arange(n) % 4,
    )
    rows = load_stage2_rows(tmp_path / "labels.npz")
    for k in ("v_target", "T", "T_succ", "cov_end", "anchor_id", "cand"):
        assert len(rows[k]) == n
    norm = Normalizer()
    norm.fit(rows["latent_stack"], rows["agent_state"])
    ve = ValueEnsemble([ValueNet() for _ in range(2)], norm)
    oe = OracleEnsemble([OracleNet() for _ in range(2)], norm)
    out = predict_rows(ve, oe, rows, np.arange(n), "cpu")
    for k in (
        "v_target",
        "T_succ",
        "cov_end",
        "anchor_id",
        "cand",
        "oracle_rm_mean_all",
        "oracle_v_mean_all",
        "v_mean_all",
        "rm_mean_all",
    ):
        assert len(out[k]) == n, k
    assert np.array_equal(out["anchor_id"], rows["seed_id"])


def test_train_member_on_epoch_callback_and_wandb_logger_noop(tmp_path):
    from train_value_model import WandbLogger, train_member

    rows = _rows(n=64)
    pose_all = pose_target(rows["cog_state"])
    norm = Normalizer()
    norm.fit(rows["latent_stack"], rows["agent_state"])
    idx = np.arange(len(rows["y"]))
    cfg_train = {
        "batch": 16,
        "mix_stage1": 0.5,
        "lr": 1e-3,
        "wd": 0.0,
        "max_epochs": 3,
        "patience": 5,
        "w_reward": 1.0,
        "w_pose": 0.5,
    }
    seen = []
    torch.manual_seed(0)
    hist, _ = train_member(
        ValueNet(), norm, rows, pose_all, idx, idx, cfg_train, "cpu", 0, "t", on_epoch=seen.append
    )
    assert seen == hist and [h["epoch"] for h in seen] == [0, 1, 2]
    # monitor split: test_loss goes to the callback only, never into history (no effect on selection)
    seen2 = []
    torch.manual_seed(0)
    hist2, best2 = train_member(
        ValueNet(),
        norm,
        rows,
        pose_all,
        idx,
        idx,
        cfg_train,
        "cpu",
        0,
        "t",
        on_epoch=seen2.append,
        monitor_idx=idx,
    )
    assert all("test_loss" in h for h in seen2) and all("test_loss" not in h for h in hist2)
    assert [h["es_loss"] for h in hist2] == [h["es_loss"] for h in hist]
    # disabled (config off / smoke / --no-wandb) -> every call is a no-op and no callback is created
    for lg in (
        WandbLogger(None, {}, "x", enabled=True),
        WandbLogger({"enable": True, "project": "p"}, {}, "x", False),
    ):
        assert lg.on_epoch_fn("value/m0") is None
        lg.summary({"a": 1})
        lg.finish()
