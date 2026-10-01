#!/usr/bin/env python
"""Step 6 of the value/reward-model plan: held-out gate report from val_predictions.npz.

Plan: value-model-opus5-5-sonnet5-floofy-engelbart.md, section "Gates". Every gate metric here uses
`v_mean_all` / `rm_mean_all` (the deterministic mean over all K members) -- the mean-of-2-random
convention (`v_mean2`) is only for search-time scoring and is not used for any metric below.
pre_success rows and wall-artifact rows are excluded from EVERY gate metric (both the pass/fail gates
and the diagnostics), per the label rules.

This script only computes numbers against the pre-registered thresholds in `cfg["gate"]`. It prints
and writes a PASS/FAIL table of whether each metric clears its threshold -- a fact about the numbers,
not a go/no-go recommendation. Opus reads gate_report.md/.json and makes that call.

`gate: {diagnostic_only: true}` (SAILOR v1, sailor_v1.md section 9 T5): G1/G2/G3/G5 are reported
without thresholds (`meets_threshold` None) and nothing fails closed; G4 and G6 keep their fixed rules.

    uv run --extra training python examples/nanowm_pusht_aug/critic/value_gate_report.py \
        --config configs/value_model_topband.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402
from gate_tables import (  # noqa: E402
    G6_BOOT,
    N_BOOT,
    diagnostic_table,
    ece,
    fast_auc as roc_auc,  # vectorised, identical values to train_value_model.roc_auc (tested)
    g6_report,
    noise_ceiling,
    noise_ceiling_subset,
    within_group_aucs,
)
from train_value_model import class_balance, load_stage2_rows, predict_rows  # noqa: E402
from value_model import OracleEnsemble, ValueEnsemble, pose_target  # noqa: E402


# ============================================================================================ metrics
def bootstrap_auroc_ci(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Grouped bootstrap 95% CI of AUROC (resample groups with replacement, recompute AUROC)."""
    if len(y) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in draw])
        a = roc_auc(y[idx], score[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_group_diff_ci(
    top_vals: np.ndarray,
    top_groups: np.ndarray,
    bot_vals: np.ndarray,
    bot_groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Grouped bootstrap 95% CI of (mean(top_vals) - mean(bot_vals)), resampling groups jointly."""
    if len(top_vals) == 0 or len(bot_vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    all_groups = np.unique(np.concatenate([top_groups, bot_groups]))
    top_by_g = {g: top_vals[top_groups == g] for g in all_groups}
    bot_by_g = {g: bot_vals[bot_groups == g] for g in all_groups}
    diffs = []
    for _ in range(n_boot):
        draw = rng.choice(all_groups, size=len(all_groups), replace=True)
        t_parts = [top_by_g[g] for g in draw if len(top_by_g[g])]
        b_parts = [bot_by_g[g] for g in draw if len(bot_by_g[g])]
        if not t_parts or not b_parts:
            continue
        t, b = np.concatenate(t_parts), np.concatenate(b_parts)
        diffs.append(float(t.mean() - b.mean()))
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(diffs))


def bootstrap_stat_ci(
    label: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Grouped bootstrap 95% CI of an arbitrary `stat_fn(label, score)` (e.g. roc_auc, or an MAE)."""
    if len(label) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in draw])
        v = stat_fn(label[idx], score[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _mae_stat(label: np.ndarray, score: np.ndarray) -> float:
    return float(np.mean(np.abs(score - label)))


def bootstrap_diff_ci_unpaired(
    label_a: np.ndarray,
    score_a: np.ndarray,
    groups_a: np.ndarray,
    label_b: np.ndarray,
    score_b: np.ndarray,
    groups_b: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """95% CI for stat_fn(a) - stat_fn(b), resampling each side's OWN groups independently -- the two
    sides don't share a group axis (e.g. stage-2 seed_ids vs stage-1 episode ids), so this is an
    UNPAIRED difference, unlike `bootstrap_group_diff_ci` above (which resamples one shared group
    axis for a paired top/bottom split of the same rows)."""
    if len(label_a) == 0 or len(label_b) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    uniq_a, uniq_b = np.unique(groups_a), np.unique(groups_b)
    idx_by_a = {g: np.nonzero(groups_a == g)[0] for g in uniq_a}
    idx_by_b = {g: np.nonzero(groups_b == g)[0] for g in uniq_b}
    diffs = []
    for _ in range(n_boot):
        draw_a = rng.choice(uniq_a, size=len(uniq_a), replace=True)
        draw_b = rng.choice(uniq_b, size=len(uniq_b), replace=True)
        idx_a = np.concatenate([idx_by_a[g] for g in draw_a])
        idx_b = np.concatenate([idx_by_b[g] for g in draw_b])
        sa = stat_fn(label_a[idx_a], score_a[idx_a])
        sb = stat_fn(label_b[idx_b], score_b[idx_b])
        if np.isfinite(sa) and np.isfinite(sb):
            diffs.append(sa - sb)
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(diffs))


def group_ids(d: dict) -> np.ndarray:
    """episode for stage-1 rows, seed_id for stage-2 rows -- same grouping as the training split."""
    return np.where(d["stage"] == 2, d["seed_id"], d["episode"]).astype(np.int64)


def gate_mask(d: dict) -> np.ndarray:
    """Rows usable for ANY gate metric: pre_success and wall-artifact rows excluded everywhere."""
    return (~d["pre_success"]) & (~d["wall_artifact"])


# ============================================================================================ gates
REQUIRED_THRESHOLDS = (
    "auroc_s1_min",  # G1
    "ece_max",  # G1 (ECE_soft against v_target)
    "auroc_s2_min",  # G2
    "within_anchor_auroc_min",  # G3
    "rm_mae_s1_max",  # G5
    "rm_mae_s2_max",  # G5
)


def _thr(gate_cfg: dict, key: str) -> float | None:
    v = gate_cfg.get(key)
    return None if v is None else float(v)


def _ok(value: float, thr: float | None, ge: bool) -> bool:
    """Fail closed: a missing threshold or a non-finite value never passes."""
    if thr is None or not np.isfinite(value):
        return False
    return bool(value >= thr) if ge else bool(value <= thr)


def compute_report(
    d: dict,
    gate_cfg: dict | None,
    test_pred: dict | None = None,
    test_skip_reason: str | None = None,
    noise_ceiling_res: dict | None = None,
    betas: tuple[float, ...] = (0.0, 0.1, 0.3),
    n_candidates: int = 8,
    n_boot_table: int = N_BOOT,
    g6_boot: int = G6_BOOT,
) -> dict:
    """All gate numbers. `gate_cfg` may be empty: thresholds that are missing make their gate FAIL
    (`meets_threshold` False, `threshold` None) and are listed in `missing_thresholds`."""
    gate_cfg = dict(gate_cfg or {})
    diagnostic_only = bool(gate_cfg.pop("diagnostic_only", False))
    if diagnostic_only:
        set_thr = [k for k in REQUIRED_THRESHOLDS if gate_cfg.get(k) is not None]
        assert not set_thr, f"gate.diagnostic_only excludes thresholds, got {set_thr}"
    missing = [] if diagnostic_only else [k for k in REQUIRED_THRESHOLDS if gate_cfg.get(k) is None]
    gm = gate_mask(d)
    groups = group_ids(d)
    report: dict = {
        "n_val_rows": int(len(d["y"])),
        "n_gate_rows": int(gm.sum()),
        "n_excluded_pre_success": int(d["pre_success"].sum()),
        "n_excluded_wall_artifact": int(d["wall_artifact"].sum()),
        "diagnostic_only": diagnostic_only,
        "missing_thresholds": missing,
    }

    # ---- G1: stage-1 RESET rows (v1 has no band bucket) ----
    m1 = gm & (d["stage"] == 1) & (d["source"] == "reset")
    auc1 = roc_auc(d["y"][m1], d["v_mean_all"][m1])
    lo1, hi1 = bootstrap_auroc_ci(d["y"][m1], d["v_mean_all"][m1], groups[m1])
    ece_soft1 = ece(d["v_target"][m1], d["v_mean_all"][m1])
    ece_bin1 = ece(d["y"][m1], d["v_mean_all"][m1])
    t_auc1, t_ece = _thr(gate_cfg, "auroc_s1_min"), _thr(gate_cfg, "ece_max")
    report["G1"] = {
        "desc": "stage-1 reset rows: V AUROC (binary y), ECE_soft (vs v_target)",
        "n": int(m1.sum()),
        "auroc": auc1,
        "auroc_ci95": [lo1, hi1],
        "ece": ece_soft1,
        "ece_soft": ece_soft1,
        "ece_bin": ece_bin1,
        "auroc_min": t_auc1,
        "ece_max": t_ece,
        "meets_threshold": _ok(auc1, t_auc1, True) and _ok(ece_soft1, t_ece, False),
    }

    # ---- G2: stage-2 imagined (r>=2) ----
    m2 = gm & (d["stage"] == 2) & (d["r"] >= 2)
    auc2 = roc_auc(d["y"][m2], d["v_mean_all"][m2])
    lo2, hi2 = bootstrap_auroc_ci(d["y"][m2], d["v_mean_all"][m2], groups[m2])
    t_auc2 = _thr(gate_cfg, "auroc_s2_min")
    report["G2"] = {
        "desc": "stage-2 imagined (r>=2): V AUROC",
        "n": int(m2.sum()),
        "auroc": auc2,
        "auroc_ci95": [lo2, hi2],
        "auroc_min": t_auc2,
        "meets_threshold": _ok(auc2, t_auc2, True),
    }
    m2_r0 = gm & (d["stage"] == 2) & (d["r"] == 0)
    report["G2_r0_diagnostic"] = {
        "desc": "stage-2 r=0 (three real context frames) -- diagnostic only, not gated",
        "n": int(m2_r0.sum()),
        "auroc": roc_auc(d["y"][m2_r0], d["v_mean_all"][m2_r0]),
    }

    # ---- G3: within-anchor ranking ----
    m3 = gm & (d["stage"] == 2)
    idx3 = np.nonzero(m3)[0]
    key3 = np.stack(
        [d["seed_id"][m3].astype(np.float64), d["r"][m3].astype(np.float64), d["sigma"][m3]], axis=1
    )
    if len(key3):
        _, inv = np.unique(key3, axis=0, return_inverse=True)
        group_aucs = within_group_aucs(d["y"][idx3], d["v_mean_all"][idx3], inv.reshape(-1))[0].tolist()
    else:
        group_aucs = []
    mean_g3 = float(np.mean(group_aucs)) if group_aucs else float("nan")
    t_g3 = _thr(gate_cfg, "within_anchor_auroc_min")
    report["G3"] = {
        "desc": "within-anchor (seed_id, r, sigma) groups, >=2 rows + mixed y: mean AUROC",
        "n_groups": len(group_aucs),
        "mean_auroc": mean_g3,
        "group_aurocs": group_aucs,
        "auroc_min": t_g3,
        "meets_threshold": _ok(mean_g3, t_g3, True),
    }

    # ---- G4: ensemble std informative (gated on |V - v_target|; binary version reported) ----
    std = d["v_std"][gm]
    g4_groups = groups[gm]
    if len(std) >= 5 and len(np.unique(std)) > 1:
        edges = np.quantile(std, [0.2, 0.4, 0.6, 0.8])
        bin_idx = np.digitize(std, edges)
        top_mask, bot_mask = bin_idx == bin_idx.max(), bin_idx == bin_idx.min()
    else:
        top_mask = bot_mask = np.zeros(len(std), bool)

    def _g4(target: np.ndarray) -> dict:
        resid = np.abs(d["v_mean_all"][gm] - target[gm].astype(np.float64))
        lo, hi, mean = bootstrap_group_diff_ci(
            resid[top_mask], g4_groups[top_mask], resid[bot_mask], g4_groups[bot_mask]
        )
        return {"mean_diff": mean, "ci95": [lo, hi]}

    g4s, g4b = _g4(d["v_target"]), _g4(d["y"])
    report["G4"] = {
        "desc": "top std quintile minus bottom: mean |V - v_target| difference, grouped bootstrap 95% CI "
        "(binary-y version reported without a threshold)",
        "n_top": int(top_mask.sum()),
        "n_bottom": int(bot_mask.sum()),
        "mean_diff": g4s["mean_diff"],
        "ci95": g4s["ci95"],
        "binary_y": g4b,
        "meets_threshold": bool(np.isfinite(g4s["ci95"][0]) and g4s["ci95"][0] > 0),
    }

    # ---- G5: RM MAE vs real coverage ----
    m5s1, m5s2 = gm & (d["stage"] == 1), gm & (d["stage"] == 2)
    mae_s1 = float(np.mean(np.abs(d["rm_mean_all"][m5s1] - d["cov"][m5s1]))) if m5s1.any() else float("nan")
    mae_s2 = float(np.mean(np.abs(d["rm_mean_all"][m5s2] - d["cov"][m5s2]))) if m5s2.any() else float("nan")
    t_m1, t_m2 = _thr(gate_cfg, "rm_mae_s1_max"), _thr(gate_cfg, "rm_mae_s2_max")
    report["G5"] = {
        "desc": "RM MAE vs real (sim) coverage",
        "stage1_n": int(m5s1.sum()),
        "stage1_mae": mae_s1,
        "stage1_mae_max": t_m1,
        "stage2_n": int(m5s2.sum()),
        "stage2_mae": mae_s2,
        "stage2_mae_max": t_m2,
        "meets_threshold": _ok(mae_s1, t_m1, False) and _ok(mae_s2, t_m2, False),
    }

    # ---- table A: 2x2 diagnostic ----
    report["table_A"] = diagnostic_table(d, gm, groups, n_boot=n_boot_table)

    # ---- G6 (+ table C) ----
    report["G6"] = g6_report(d, test_pred, list(betas), n_candidates, test_skip_reason, n_boot=g6_boot)

    # ---- table D ----
    nc = (
        noise_ceiling_res
        if noise_ceiling_res is not None
        else {"status": "skipped", "reason": "no re-label files"}
    )
    report["table_D"] = nc
    if nc.get("status") == "ok" and t_g3 is not None:
        report["table_D"]["g3_threshold_below_ceiling"] = bool(t_g3 < nc["within_group_auroc"])

    # ---- diagnostics (no threshold) ----
    diag: dict = {}
    for stage in (1, 2):
        ms = gm & (d["stage"] == stage)
        diag[f"latent_vs_oracle_stage{stage}"] = {
            "n": int(ms.sum()),
            "latent_auroc": roc_auc(d["y"][ms], d["v_mean_all"][ms]),
            "oracle_auroc": roc_auc(d["y"][ms], d["oracle_v_mean_all"][ms]),
        }
    diag["auroc_by_r"] = {
        str(int(r)): {
            "n": int((gm & (d["stage"] == 2) & (d["r"] == r)).sum()),
            "auroc": roc_auc(
                d["y"][gm & (d["stage"] == 2) & (d["r"] == r)],
                d["v_mean_all"][gm & (d["stage"] == 2) & (d["r"] == r)],
            ),
        }
        for r in sorted(set(d["r"][gm & (d["stage"] == 2)].tolist()))
    }
    sigma_round = np.round(d["sigma"], 3)
    diag["auroc_by_sigma_stage2"] = {
        str(float(s)): {
            "n": int((gm & (d["stage"] == 2) & (sigma_round == s)).sum()),
            "auroc": roc_auc(
                d["y"][gm & (d["stage"] == 2) & (sigma_round == s)],
                d["v_mean_all"][gm & (d["stage"] == 2) & (sigma_round == s)],
            ),
        }
        for s in sorted(set(sigma_round[gm & (d["stage"] == 2)].tolist()))
    }
    m_pose = gm & (d["stage"] == 1)
    if m_pose.any():
        true_pose, pred_pose = pose_target(d["cog_state"][m_pose]), d["pose_pred"][m_pose]
        true_th = np.arctan2(true_pose[:, 2], true_pose[:, 3])
        pred_th = np.arctan2(pred_pose[:, 2], pred_pose[:, 3])
        ang_err = np.abs((true_th - pred_th + np.pi) % (2 * np.pi) - np.pi)
        diag["pose_angle_err_deg_median"] = float(np.degrees(np.median(ang_err)))
    else:
        diag["pose_angle_err_deg_median"] = float("nan")
    diag["class_balance_gate_rows"] = class_balance(d["y"][gm], d["source"][gm])
    report["diagnostics"] = diag

    report["real_vs_imagined"] = real_vs_imagined_block(d, gm, groups)

    # ---- fail-closed summary: G1-G5 need every threshold; G6 is evaluated on test anchors only ----
    if diagnostic_only:
        for k in ("G1", "G2", "G3", "G5"):
            report[k]["meets_threshold"] = None
        report["g1_g5_all_meet_threshold"] = None
        report["fail_closed"] = False
        return report
    g15 = [report[k]["meets_threshold"] for k in ("G1", "G2", "G3", "G4", "G5")]
    report["g1_g5_all_meet_threshold"] = bool(not missing and all(g15))
    report["fail_closed"] = bool(missing)
    return report


# ==================================================================================== real vs imagined
def _rvi_cell(d: dict, mask: np.ndarray, groups: np.ndarray, seed: int, prefix: str = "") -> dict:
    """AUROC(V), AUROC(RM as a success score) and RM MAE vs its coverage label, each with a grouped
    (episode / seed_id) bootstrap 95% CI. `mask` must already have pre_success/wall_artifact excluded."""
    y, v, rm, cov, g = (
        d["y"][mask],
        d[prefix + "v_mean_all"][mask],
        d[prefix + "rm_mean_all"][mask],
        d["cov"][mask],
        groups[mask],
    )
    auc_v = roc_auc(y, v)
    auc_v_lo, auc_v_hi = bootstrap_stat_ci(y, v, g, roc_auc, seed=seed)
    auc_rm = roc_auc(y, rm)
    auc_rm_lo, auc_rm_hi = bootstrap_stat_ci(y, rm, g, roc_auc, seed=seed + 1)
    mae = _mae_stat(cov, rm) if mask.any() else float("nan")
    mae_lo, mae_hi = bootstrap_stat_ci(cov, rm, g, _mae_stat, seed=seed + 2)
    return {
        "n": int(mask.sum()),
        "n_groups": int(len(np.unique(g))) if mask.any() else 0,
        "auroc_v": auc_v,
        "auroc_v_ci95": [auc_v_lo, auc_v_hi],
        "auroc_rm": auc_rm,
        "auroc_rm_ci95": [auc_rm_lo, auc_rm_hi],
        "rm_mae": mae,
        "rm_mae_ci95": [mae_lo, mae_hi],
    }


def real_vs_imagined_block(d: dict, gm: np.ndarray, groups: np.ndarray) -> dict:
    """Diagnostic only (no threshold, doesn't touch G1-G5): are V and RM useful on imagined (stage-2,
    r>=2) rollouts, same as on real (stage-1) ones? pre_success/wall_artifact already excluded via
    `gm`. Real is split band/reset (stage 1's own two sources); imagined is stage-2 r>=2; stage-2 r=0
    (three real context frames) is reported separately, since it isn't really "imagined"."""
    m_real_band = gm & (d["stage"] == 1) & (d["source"] == "band")
    m_real_reset = gm & (d["stage"] == 1) & (d["source"] == "reset")
    m_imagined = gm & (d["stage"] == 2) & (d["r"] >= 2)
    m_s2_r0 = gm & (d["stage"] == 2) & (d["r"] == 0)
    masks = {
        "real_band": (m_real_band, 5000),
        "real_reset": (m_real_reset, 5010),
        "imagined_r_ge2": (m_imagined, 5020),
        "stage2_r0_diagnostic": (m_s2_r0, 5030),
    }
    cells = {k: _rvi_cell(d, m, groups, seed=sd) for k, (m, sd) in masks.items()}
    # OracleNet rows (true 8-dim state): same rows, same bootstrap seeds
    oracle = {
        f"oracle_{k}": _rvi_cell(d, m, groups, seed=sd, prefix="oracle_")
        for k, (m, sd) in masks.items()
        if k != "real_band"
    }

    # UNPAIRED imagined-vs-real gap; the real reference is stage-1 reset (v1 has no band bucket)
    def _gap(key_label: str, key_score: str, stat, seed: int) -> dict:
        lo, hi, diff = bootstrap_diff_ci_unpaired(
            d[key_label][m_imagined],
            d[key_score][m_imagined],
            groups[m_imagined],
            d[key_label][m_real_reset],
            d[key_score][m_real_reset],
            groups[m_real_reset],
            stat,
            seed=seed,
        )
        return {"mean_diff": diff, "ci95": [lo, hi]}

    unpaired = {
        "desc": "imagined (stage-2 r>=2) minus real (stage-1 reset), UNPAIRED -- different row "
        "populations (time since start, perturbation schedule), so this is a diagnostic gap, not "
        "attributable to imagination alone.",
        "auroc_v": _gap("y", "v_mean_all", roc_auc, 5100),
        "auroc_rm": _gap("y", "rm_mean_all", roc_auc, 5101),
        "rm_mae": _gap("cov", "rm_mean_all", _mae_stat, 5102),
        "oracle_auroc_v": _gap("y", "oracle_v_mean_all", roc_auc, 5103),
        "oracle_auroc_rm": _gap("y", "oracle_rm_mean_all", roc_auc, 5104),
        "oracle_rm_mae": _gap("cov", "oracle_rm_mean_all", _mae_stat, 5105),
    }

    return {
        "desc": "Diagnostic only (no threshold): are V/RM useful on imagined rows, same as on real "
        "rows? pre_success and wall_artifact rows excluded, as everywhere else.",
        **cells,
        **oracle,
        "imagined_minus_real_reset_unpaired": unpaired,
    }


# ============================================================================================ output
def write_plots(d: dict, report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gm = gate_mask(d)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # reliability diagram (G1 rows = stage-1 reset if any, else all gate rows); y axis = mean v_target
    m1 = gm & (d["stage"] == 1) & (d["source"] == "reset")
    rel_mask = m1 if m1.any() else gm
    y, prob = d["v_target"][rel_mask].astype(np.float64), d["v_mean_all"][rel_mask]
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.any():
            xs.append(prob[m].mean())
            ys.append(y[m].mean())
            ns.append(int(m.sum()))
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    axes[0].scatter(xs, ys, s=[20 + 4 * n for n in ns])
    axes[0].set_xlabel("mean predicted V")
    axes[0].set_ylabel("mean v_target")
    axes[0].set_title(f"reliability (n={int(rel_mask.sum())})")
    axes[0].legend(fontsize=8)

    by_r = report["diagnostics"]["auroc_by_r"]
    rs = sorted(by_r, key=int)
    axes[1].bar(rs, [by_r[r]["auroc"] for r in rs])
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("AUROC by round r (stage 2)")
    axes[1].set_xlabel("r")
    axes[1].set_ylim(0, 1)

    by_sig = report["diagnostics"]["auroc_by_sigma_stage2"]
    sigs = sorted(by_sig, key=float)
    axes[2].bar(sigs, [by_sig[s]["auroc"] for s in sigs])
    axes[2].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("AUROC by inject sigma (stage 2)")
    axes[2].set_xlabel("sigma")
    axes[2].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(x) -> str:
    return "None" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def _ci(c) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def to_markdown(report: dict) -> str:
    lines = ["# Value/RM gate report", "", "Metrics only. Opus makes the go/no-go call.", ""]
    if report.get("diagnostic_only"):
        lines += [
            "**Diagnostic-only mode: G1/G2/G3/G5 have no thresholds (`meets threshold` = None). "
            "The decision gate is G6 (offline Best-of-N on test anchors).**",
            "",
        ]
    if report["missing_thresholds"]:
        lines += [
            f"**FAIL CLOSED: gate thresholds missing in config ({', '.join(report['missing_thresholds'])}). "
            "Every gate that needs them is reported as NOT met.**",
            "",
        ]
    lines += [
        f"- val rows: {report['n_val_rows']} total, {report['n_gate_rows']} usable for gate metrics "
        f"({report['n_excluded_pre_success']} pre_success + {report['n_excluded_wall_artifact']} "
        "wall_artifact excluded, may overlap)",
        "",
        "| gate | metric | value | threshold | meets threshold |",
        "| --- | --- | --- | --- | --- |",
    ]
    g1 = report["G1"]
    lines.append(
        f"| G1 | stage-1 reset AUROC (n={g1['n']}) | {g1['auroc']:.4f} (CI {g1['auroc_ci95']}) "
        f"| >= {_fmt(g1['auroc_min'])} | {g1['meets_threshold']} |"
    )
    lines.append(
        f"| G1 | stage-1 reset ECE_soft (ECE_bin {g1['ece_bin']:.4f}, no threshold) | {g1['ece_soft']:.4f} "
        f"| <= {_fmt(g1['ece_max'])} | {g1['meets_threshold']} |"
    )
    g2 = report["G2"]
    lines.append(
        f"| G2 | stage-2 r>=2 AUROC (n={g2['n']}) | {g2['auroc']:.4f} (CI {g2['auroc_ci95']}) "
        f"| >= {_fmt(g2['auroc_min'])} | {g2['meets_threshold']} |"
    )
    g3 = report["G3"]
    lines.append(
        f"| G3 | within-anchor mean AUROC ({g3['n_groups']} groups) | {g3['mean_auroc']:.4f} "
        f"| >= {_fmt(g3['auroc_min'])} | {g3['meets_threshold']} |"
    )
    g4 = report["G4"]
    lines.append(
        f"| G4 | top-bottom std quintile \\|V-v_target\\| diff (n_top={g4['n_top']}, n_bot={g4['n_bottom']}) "
        f"| {g4['mean_diff']:.4f} (CI {g4['ci95']}) | CI lower > 0 | {g4['meets_threshold']} |"
    )
    g5 = report["G5"]
    lines.append(
        f"| G5 | stage-1 RM MAE (n={g5['stage1_n']}) | {g5['stage1_mae']:.4f} | <= {_fmt(g5['stage1_mae_max'])} | {g5['meets_threshold']} |"
    )
    lines.append(
        f"| G5 | stage-2 RM MAE (n={g5['stage2_n']}) | {g5['stage2_mae']:.4f} | <= {_fmt(g5['stage2_mae_max'])} | {g5['meets_threshold']} |"
    )
    g6 = report["G6"]
    lines.append(
        f"| G6 | offline Best-of-N on test anchors | status: {g6['status']} | one-sided 95% lower bound of "
        f"mean delta > 0 | {g6['meets_threshold']} |"
    )
    lines += [
        "",
        f"G1-G5 all meet threshold: **{report['g1_g5_all_meet_threshold']}**"
        + (" (not gated in diagnostic-only mode)" if report.get("diagnostic_only") else ""),
        "",
        f"Diagnostic: stage-2 r=0 AUROC (n={report['G2_r0_diagnostic']['n']}) = {report['G2_r0_diagnostic']['auroc']:.4f}",
        "",
    ]

    ta = report["table_A"]
    lines += ["## Table A: 2x2 diagnostic (val, gate rows)", "", ta["desc"], ""]
    lines.append("| cell | n | n groups | AUROC | ECE_soft | within-anchor AUROC | RM MAE |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for col, cname in (
        ("s1", "stage-1 real"),
        ("s2", "stage-2 r>=2 imagined"),
        ("s2_r0", "stage-2 r=0 (real frames)"),
    ):
        c = ta["cells"][col]
        for m, mname in (("value", "ValueNet"), ("oracle", "OracleNet")):
            cells = " | ".join(
                f"{c[m][k]['value']:.3f} {_ci(c[m][k]['ci95'])}"
                for k in ("auroc", "ece_soft", "within_anchor_auroc", "rm_mae")
            )
            lines.append(f"| {mname} / {cname} | {c['n']} | {c['n_groups']} | {cells} |")
    lines += [
        "",
        "| delta | AUROC | ECE_soft | within-anchor AUROC | RM MAE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dk, dname in (
        ("c_minus_a", "c-a (encoder loss)"),
        ("d_minus_b", "d-b (encoder + WM)"),
        ("dmb_minus_cma", "(d-b)-(c-a) (WM hallucination)"),
        ("c_minus_d", "c-d (imagined harder)"),
    ):
        row = " | ".join(
            f"{ta['deltas'][k][dk]['value']:.3f} {_ci(ta['deltas'][k][dk]['ci95'])}"
            for k in ("auroc", "ece_soft", "within_anchor_auroc", "rm_mae")
        )
        lines.append(f"| {dname} | {row} |")
    lines.append("")

    lines += ["## Table C / G6: offline Best-of-N (r=2 query)", "", g6["desc"], ""]
    lines.append(f"- status: **{g6['status']}**" + (f" -- {g6.get('reason')}" if g6.get("reason") else ""))
    lines.append(f"- selected beta (on val): {g6['selected_beta']}")
    for split in ("val", "test"):
        r = g6.get(split)
        if not r:
            continue
        lines.append(
            f"- {split}: n_used={r['n_used']}, n_informative={r['n_informative']}, wall-excluded={r['n_excluded_wall']}, "
            f"missing-excluded={r['n_excluded_missing']}, underpowered={r['underpowered']}"
        )
        lines.append("")
        lines.append(
            "| model / beta | S_random | S_selected | S_hindsight | mean delta [95% CI] | SD(delta) | one-sided 95% lower | gap closed [CI] |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for b, pb in r["per_beta"].items():
            for m, mname in (("value", "ValueNet"), ("oracle", "OracleNet")):
                s_ = pb[m]
                lines.append(
                    f"| {split} {mname} beta={b} | {s_['S_random']:.3f} | {s_['S_sel']:.3f} | {s_['S_best']:.3f} "
                    f"| {s_['delta_mean']:.3f} {_ci(s_['delta_ci95'])} | {s_['delta_sd']:.3f} "
                    f"| {s_['delta_lower_one_sided_95']:.3f} | {s_['gap_closed']:.3f} {_ci(s_['gap_closed_ci95'])} |"
                )
        lines.append("")

    nc = report["table_D"]
    lines += ["## Table D: noise ceiling", ""]
    if nc.get("status") == "ok":
        lines.append(
            f"- rows {nc['n_rows']}, anchors {nc['n_anchors']}, mixed groups {nc['n_groups_mixed']}: y agreement "
            f"{nc['y_agreement']:.3f} {_ci(nc['y_agreement_ci95'])}, within-group AUROC(y1, y2) "
            f"{nc['within_group_auroc']:.3f} {_ci(nc['within_group_auroc_ci95'])}"
        )
        if "g3_threshold_below_ceiling" in nc:
            lines.append(f"- G3 threshold below the ceiling: {nc['g3_threshold_below_ceiling']}")
    else:
        lines.append(f"- {nc.get('status')}: {nc.get('reason', '')}")
    lines.append("")

    d = report["diagnostics"]
    lines.append("## Diagnostics (no threshold)")
    for stage in (1, 2):
        lv = d[f"latent_vs_oracle_stage{stage}"]
        lines.append(
            f"- stage {stage}: latent AUROC {lv['latent_auroc']:.4f} vs oracle AUROC {lv['oracle_auroc']:.4f} (n={lv['n']})"
        )
    lines.append(f"- AUROC by r (stage 2): {d['auroc_by_r']}")
    lines.append(f"- AUROC by sigma (stage 2): {d['auroc_by_sigma_stage2']}")
    lines.append(f"- pose-aux angle error, median (deg, stage 1): {d['pose_angle_err_deg_median']:.2f}")
    lines.append(f"- class balance (gate rows): {d['class_balance_gate_rows']}")
    lines.append("")

    rvi = report["real_vs_imagined"]
    lines.append("## Real vs imagined (diagnostic only, no threshold)")
    lines.append(rvi["desc"])
    lines.append("")
    lines.append("| rows | n | n groups | AUROC(V) [CI] | AUROC(RM) [CI] | RM MAE [CI] |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in (
        ("real_band", "ValueNet, real, stage-1 band"),
        ("real_reset", "ValueNet, real, stage-1 reset"),
        ("imagined_r_ge2", "ValueNet, imagined, stage-2 r>=2"),
        ("stage2_r0_diagnostic", "ValueNet, stage-2 r=0 (3 real frames)"),
        ("oracle_real_reset", "OracleNet, real, stage-1 reset"),
        ("oracle_imagined_r_ge2", "OracleNet, imagined, stage-2 r>=2"),
        ("oracle_stage2_r0_diagnostic", "OracleNet, stage-2 r=0"),
    ):
        c = rvi[key]
        lines.append(
            f"| {label} | {c['n']} | {c['n_groups']} | {c['auroc_v']:.4f} {c['auroc_v_ci95']} "
            f"| {c['auroc_rm']:.4f} {c['auroc_rm_ci95']} | {c['rm_mae']:.4f} {c['rm_mae_ci95']} |"
        )
    up = rvi["imagined_minus_real_reset_unpaired"]
    lines.append("")
    lines.append(up["desc"])
    for k in ("auroc_v", "auroc_rm", "rm_mae", "oracle_auroc_v", "oracle_auroc_rm", "oracle_rm_mae"):
        lines.append(f"- {k} diff: {up[k]['mean_diff']:.4f} (CI {up[k]['ci95']})")
    lines.append("")
    lines.append("See gate_report.png for the reliability diagram and AUROC-by-r/sigma plots.")
    return "\n".join(lines)


def load_test_predictions(
    cfg: dict, model_dir: Path, test_root: Path, device: str
) -> tuple[dict | None, str | None]:
    """Predictions of the saved ensembles on the G6 test anchors. (None, reason) when the test labels /
    models do not exist yet -- G6 is then skipped, never an error."""
    labels = test_root / "labels.npz"
    if not labels.exists():
        return None, f"test anchor labels not found ({labels}); run test_anchors/test_imagine/test_label (T7)"
    for f in ("value_ensemble.pt", "oracle_ensemble.pt"):
        if not (model_dir / f).exists():
            return None, f"missing {model_dir / f}; run the train stage first"
    rows = load_stage2_rows(labels)
    assert (rows["anchor_id"] >= 0).all()
    ve = ValueEnsemble.load(model_dir / "value_ensemble.pt", device=device)
    oe = OracleEnsemble.load(model_dir / "oracle_ensemble.pt", device=device)
    return predict_rows(ve, oe, rows, np.arange(len(rows["y"])), device), None


def load_noise_ceiling(stage2_root: Path, names: tuple[str, str]) -> dict:
    paths = [stage2_root / n for n in names]
    if not all(p.exists() for p in paths):
        return {
            "status": "skipped",
            "reason": f"re-label files not found: {[str(p) for p in paths if not p.exists()]}",
        }
    return noise_ceiling(load_stage2_rows(paths[0]), load_stage2_rows(paths[1]))


NC_NAMES = (
    "labels_nc0.npz",
    "labels_nc1.npz",
)  # `label --anchor-subset ids.npz --cont-seed 770000|770001 --out-name`


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--nc-subset-out",
        type=Path,
        default=None,
        help="only write the Table D anchor ids (seed ids from fit+es splits, label.noise_ceiling) and exit",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["out_dir"])
    base = root / "smoke" if args.smoke else root
    model_dir = base / "model"
    report_dir = base
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.nc_subset_out is not None:
        ncfg = cfg["label"]["noise_ceiling"]
        info = json.loads((model_dir / "split.json").read_text())["buckets"]["stage2"]
        ids = noise_ceiling_subset(
            info["fit_groups"],
            info["es_groups"],
            ncfg["n_anchors"],
            ncfg["sample_seed"],
            tuple(ncfg["split"]),
        )
        np.savez(args.nc_subset_out, seed_id=ids)
        print(f"wrote {len(ids)} Table D anchors -> {args.nc_subset_out}")
        return

    pred_path = model_dir / "val_predictions.npz"
    assert pred_path.exists(), f"missing {pred_path} -- run the train stage first"
    d = dict(np.load(pred_path, allow_pickle=True))
    print(f"loaded {len(d['y'])} val rows <- {pred_path}")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_pred, skip_reason = load_test_predictions(cfg, model_dir, base / "test_stage2", device)
    if test_pred is None:
        print(f"G6 test anchors SKIPPED: {skip_reason}")
    k = cfg["smoke"]["stage2_k"] if args.smoke else cfg["stage2"]["k"]
    n_cand = int(k) * len(cfg["stage2"]["inject_sigma"])
    betas = tuple(float(b) for b in cfg["label"]["beta_candidates"])
    nc = load_noise_ceiling(base / "stage2", NC_NAMES)

    gate_cfg = cfg.get("gate") or {}
    report = compute_report(d, gate_cfg, test_pred, skip_reason, nc, betas=betas, n_candidates=n_cand)
    (report_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (report_dir / "gate_report.md").write_text(md)
    write_plots(d, report, report_dir / "gate_report.png")

    print(
        f"\nwrote {report_dir / 'gate_report.json'}, {report_dir / 'gate_report.md'}, {report_dir / 'gate_report.png'}\n"
    )
    for k_ in ("G1", "G2", "G3", "G4", "G5", "G6"):
        g = report[k_]
        print(f"  {k_}: {g.get('desc', '')} -> meets_threshold={g['meets_threshold']}")
    print("\n(Opus decides go/no-go from gate_report.md; this table is arithmetic only.)")
    if report["fail_closed"]:
        print(f"FAIL CLOSED: missing gate thresholds {report['missing_thresholds']} -> exit 2")
        sys.exit(2)


if __name__ == "__main__":
    main()
