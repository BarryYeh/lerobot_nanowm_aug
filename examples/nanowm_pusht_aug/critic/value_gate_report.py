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
from train_value_model import class_balance, roc_auc  # noqa: E402  (both pure numpy, no heavy deps)
from value_model import pose_target  # noqa: E402


# ============================================================================================ metrics
def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, 10 equal-width bins on [0,1]."""
    y_true, prob = np.asarray(y_true, np.float64), np.asarray(prob, np.float64)
    if len(prob) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    total, e = len(prob), 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        e += (m.sum() / total) * abs(y_true[m].mean() - prob[m].mean())
    return float(e)


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
def compute_report(d: dict, gate_cfg: dict) -> dict:
    gm = gate_mask(d)
    groups = group_ids(d)
    report: dict = {
        "n_val_rows": int(len(d["y"])),
        "n_gate_rows": int(gm.sum()),
        "n_excluded_pre_success": int(d["pre_success"].sum()),
        "n_excluded_wall_artifact": int(d["wall_artifact"].sum()),
    }

    # ---- G1: stage-1 band rows ----
    m1 = gm & (d["stage"] == 1) & (d["source"] == "band")
    auc1 = roc_auc(d["y"][m1], d["v_mean_all"][m1])
    lo1, hi1 = bootstrap_auroc_ci(d["y"][m1], d["v_mean_all"][m1], groups[m1])
    ece1 = ece(d["y"][m1], d["v_mean_all"][m1])
    g1_pass = bool(np.isfinite(auc1) and auc1 >= gate_cfg["auroc_s1_min"] and ece1 <= gate_cfg["ece_max"])
    report["G1"] = {
        "desc": "stage-1 band: V AUROC, ECE",
        "n": int(m1.sum()),
        "auroc": auc1,
        "auroc_ci95": [lo1, hi1],
        "ece": ece1,
        "auroc_min": gate_cfg["auroc_s1_min"],
        "ece_max": gate_cfg["ece_max"],
        "meets_threshold": g1_pass,
    }

    # ---- G2: stage-2 imagined (r>=2) ----
    m2 = gm & (d["stage"] == 2) & (d["r"] >= 2)
    auc2 = roc_auc(d["y"][m2], d["v_mean_all"][m2])
    lo2, hi2 = bootstrap_auroc_ci(d["y"][m2], d["v_mean_all"][m2], groups[m2])
    g2_pass = bool(np.isfinite(auc2) and auc2 >= gate_cfg["auroc_s2_min"])
    report["G2"] = {
        "desc": "stage-2 imagined (r>=2): V AUROC",
        "n": int(m2.sum()),
        "auroc": auc2,
        "auroc_ci95": [lo2, hi2],
        "auroc_min": gate_cfg["auroc_s2_min"],
        "meets_threshold": g2_pass,
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
    uniq_keys, inv = (
        np.unique(key3, axis=0, return_inverse=True) if len(key3) else (np.zeros((0, 3)), np.array([]))
    )
    group_aucs = []
    for gi in range(len(uniq_keys)):
        sel = inv == gi
        yy = d["y"][idx3[sel]]
        if sel.sum() < 2 or len(np.unique(yy)) < 2:
            continue
        a = roc_auc(yy, d["v_mean_all"][idx3[sel]])
        if np.isfinite(a):
            group_aucs.append(a)
    mean_g3 = float(np.mean(group_aucs)) if group_aucs else float("nan")
    g3_pass = bool(np.isfinite(mean_g3) and mean_g3 >= gate_cfg["within_anchor_auroc_min"])
    report["G3"] = {
        "desc": "within-anchor (seed_id, r, sigma) groups, >=2 rows + mixed y: mean AUROC",
        "n_groups": len(group_aucs),
        "mean_auroc": mean_g3,
        "group_aurocs": group_aucs,
        "auroc_min": gate_cfg["within_anchor_auroc_min"],
        "meets_threshold": g3_pass,
    }

    # ---- G4: ensemble std informative ----
    std = d["v_std"][gm]
    absresid = np.abs(d["v_mean_all"][gm] - d["y"][gm].astype(np.float64))
    g4_groups = groups[gm]
    if len(std) >= 5 and len(np.unique(std)) > 1:
        edges = np.quantile(std, [0.2, 0.4, 0.6, 0.8])
        bin_idx = np.digitize(std, edges)
        top_mask, bot_mask = bin_idx == bin_idx.max(), bin_idx == bin_idx.min()
    else:
        top_mask = bot_mask = np.zeros(len(std), bool)
    lo4, hi4, mean4 = bootstrap_group_diff_ci(
        absresid[top_mask], g4_groups[top_mask], absresid[bot_mask], g4_groups[bot_mask]
    )
    g4_pass = bool(np.isfinite(lo4) and lo4 > 0)
    report["G4"] = {
        "desc": "top std quintile minus bottom: mean |V-y| difference, grouped bootstrap 95% CI",
        "n_top": int(top_mask.sum()),
        "n_bottom": int(bot_mask.sum()),
        "mean_diff": mean4,
        "ci95": [lo4, hi4],
        "meets_threshold": g4_pass,
    }

    # ---- G5: RM MAE vs coverage ----
    m5s1, m5s2 = gm & (d["stage"] == 1), gm & (d["stage"] == 2)
    mae_s1 = float(np.mean(np.abs(d["rm_mean_all"][m5s1] - d["cov"][m5s1]))) if m5s1.any() else float("nan")
    mae_s2 = float(np.mean(np.abs(d["rm_mean_all"][m5s2] - d["cov"][m5s2]))) if m5s2.any() else float("nan")
    g5_pass = bool(
        np.isfinite(mae_s1)
        and np.isfinite(mae_s2)
        and mae_s1 <= gate_cfg["rm_mae_s1_max"]
        and mae_s2 <= gate_cfg["rm_mae_s2_max"]
    )
    report["G5"] = {
        "desc": "RM MAE vs sim/imagined coverage",
        "stage1_n": int(m5s1.sum()),
        "stage1_mae": mae_s1,
        "stage1_mae_max": gate_cfg["rm_mae_s1_max"],
        "stage2_n": int(m5s2.sum()),
        "stage2_mae": mae_s2,
        "stage2_mae_max": gate_cfg["rm_mae_s2_max"],
        "meets_threshold": g5_pass,
    }

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
    return report


# ==================================================================================== real vs imagined
def _rvi_cell(d: dict, mask: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """AUROC(V), AUROC(RM as a success score) and RM MAE vs its coverage label, each with a grouped
    (episode / seed_id) bootstrap 95% CI. `mask` must already have pre_success/wall_artifact excluded."""
    y, v, rm, cov, g = (
        d["y"][mask],
        d["v_mean_all"][mask],
        d["rm_mean_all"][mask],
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

    real_band = _rvi_cell(d, m_real_band, groups, seed=5000)
    real_reset = _rvi_cell(d, m_real_reset, groups, seed=5010)
    imagined = _rvi_cell(d, m_imagined, groups, seed=5020)
    s2_r0 = _rvi_cell(d, m_s2_r0, groups, seed=5030)

    auc_v_lo, auc_v_hi, auc_v_diff = bootstrap_diff_ci_unpaired(
        d["y"][m_imagined],
        d["v_mean_all"][m_imagined],
        groups[m_imagined],
        d["y"][m_real_band],
        d["v_mean_all"][m_real_band],
        groups[m_real_band],
        roc_auc,
        seed=5100,
    )
    auc_rm_lo, auc_rm_hi, auc_rm_diff = bootstrap_diff_ci_unpaired(
        d["y"][m_imagined],
        d["rm_mean_all"][m_imagined],
        groups[m_imagined],
        d["y"][m_real_band],
        d["rm_mean_all"][m_real_band],
        groups[m_real_band],
        roc_auc,
        seed=5101,
    )
    mae_lo, mae_hi, mae_diff = bootstrap_diff_ci_unpaired(
        d["cov"][m_imagined],
        d["rm_mean_all"][m_imagined],
        groups[m_imagined],
        d["cov"][m_real_band],
        d["rm_mean_all"][m_real_band],
        groups[m_real_band],
        _mae_stat,
        seed=5102,
    )
    unpaired = {
        "desc": "imagined (stage-2 r>=2) minus real (stage-1 band), UNPAIRED -- different row "
        "populations (time since start, perturbation schedule), so this is a diagnostic gap, not "
        "attributable to imagination alone.",
        "auroc_v": {"mean_diff": auc_v_diff, "ci95": [auc_v_lo, auc_v_hi]},
        "auroc_rm": {"mean_diff": auc_rm_diff, "ci95": [auc_rm_lo, auc_rm_hi]},
        "rm_mae": {"mean_diff": mae_diff, "ci95": [mae_lo, mae_hi]},
    }

    return {
        "desc": "Diagnostic only (no threshold): are V/RM useful on imagined rows, same as on real "
        "rows? pre_success and wall_artifact rows excluded, as everywhere else.",
        "real_band": real_band,
        "real_reset": real_reset,
        "imagined_r_ge2": imagined,
        "stage2_r0_diagnostic": s2_r0,
        "imagined_minus_real_band_unpaired": unpaired,
    }


# ============================================================================================ output
def write_plots(d: dict, report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gm = gate_mask(d)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # reliability diagram (G1 rows if any, else all gate rows)
    m1 = gm & (d["stage"] == 1) & (d["source"] == "band")
    rel_mask = m1 if m1.any() else gm
    y, prob = d["y"][rel_mask].astype(np.float64), d["v_mean_all"][rel_mask]
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
    axes[0].set_ylabel("observed success rate")
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


def to_markdown(report: dict) -> str:
    lines = ["# Value/RM gate report", "", "Metrics only. Opus makes the go/no-go call.", ""]
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
        f"| G1 | stage-1 band AUROC (n={g1['n']}) | {g1['auroc']:.4f} (CI {g1['auroc_ci95']}) "
        f"| >= {g1['auroc_min']} | {g1['meets_threshold']} |"
    )
    lines.append(
        f"| G1 | stage-1 band ECE | {g1['ece']:.4f} | <= {g1['ece_max']} | {g1['meets_threshold']} |"
    )
    g2 = report["G2"]
    lines.append(
        f"| G2 | stage-2 r>=2 AUROC (n={g2['n']}) | {g2['auroc']:.4f} (CI {g2['auroc_ci95']}) "
        f"| >= {g2['auroc_min']} | {g2['meets_threshold']} |"
    )
    g3 = report["G3"]
    lines.append(
        f"| G3 | within-anchor mean AUROC ({g3['n_groups']} groups) | {g3['mean_auroc']:.4f} "
        f"| >= {g3['auroc_min']} | {g3['meets_threshold']} |"
    )
    g4 = report["G4"]
    lines.append(
        f"| G4 | top-bottom std quintile \\|V-y\\| diff (n_top={g4['n_top']}, n_bot={g4['n_bottom']}) "
        f"| {g4['mean_diff']:.4f} (CI {g4['ci95']}) | CI lower > 0 | {g4['meets_threshold']} |"
    )
    g5 = report["G5"]
    lines.append(
        f"| G5 | stage-1 RM MAE (n={g5['stage1_n']}) | {g5['stage1_mae']:.4f} | <= {g5['stage1_mae_max']} | {g5['meets_threshold']} |"
    )
    lines.append(
        f"| G5 | stage-2 RM MAE (n={g5['stage2_n']}) | {g5['stage2_mae']:.4f} | <= {g5['stage2_mae_max']} | {g5['meets_threshold']} |"
    )
    lines += [
        "",
        f"Diagnostic: stage-2 r=0 AUROC (n={report['G2_r0_diagnostic']['n']}) = {report['G2_r0_diagnostic']['auroc']:.4f}",
        "",
    ]
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
        ("real_band", "real, stage-1 band"),
        ("real_reset", "real, stage-1 reset"),
        ("imagined_r_ge2", "imagined, stage-2 r>=2"),
        ("stage2_r0_diagnostic", "stage-2 r=0 (3 real frames)"),
    ):
        c = rvi[key]
        lines.append(
            f"| {label} | {c['n']} | {c['n_groups']} | {c['auroc_v']:.4f} {c['auroc_v_ci95']} "
            f"| {c['auroc_rm']:.4f} {c['auroc_rm_ci95']} | {c['rm_mae']:.4f} {c['rm_mae_ci95']} |"
        )
    up = rvi["imagined_minus_real_band_unpaired"]
    lines.append("")
    lines.append(up["desc"])
    lines.append(
        f"- AUROC(V) diff: {up['auroc_v']['mean_diff']:.4f} (CI {up['auroc_v']['ci95']})\n"
        f"- AUROC(RM) diff: {up['auroc_rm']['mean_diff']:.4f} (CI {up['auroc_rm']['ci95']})\n"
        f"- RM MAE diff: {up['rm_mae']['mean_diff']:.4f} (CI {up['rm_mae']['ci95']})"
    )
    lines.append("")
    lines.append("See gate_report.png for the reliability diagram and AUROC-by-r/sigma plots.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["out_dir"])
    model_dir = root / ("smoke/model" if args.smoke else "model")
    report_dir = root / "smoke" if args.smoke else root
    report_dir.mkdir(parents=True, exist_ok=True)

    pred_path = model_dir / "val_predictions.npz"
    assert pred_path.exists(), f"missing {pred_path} -- run the train stage first"
    d = dict(np.load(pred_path, allow_pickle=True))
    print(f"loaded {len(d['y'])} val rows <- {pred_path}")

    report = compute_report(d, cfg["gate"])
    (report_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (report_dir / "gate_report.md").write_text(md)
    write_plots(d, report, report_dir / "gate_report.png")

    print(
        f"\nwrote {report_dir / 'gate_report.json'}, {report_dir / 'gate_report.md'}, {report_dir / 'gate_report.png'}\n"
    )
    print("| gate | value | threshold | meets_threshold |")
    for k in ("G1", "G2", "G3", "G4", "G5"):
        g = report[k]
        print(f"  {k}: {g.get('desc', '')} -> meets_threshold={g['meets_threshold']}")
    print("\n(Opus decides go/no-go from gate_report.md; this table is arithmetic only.)")


if __name__ == "__main__":
    main()
