#!/usr/bin/env python
"""SAILOR expand S6: score every branch with the value models and grade them against the 20-pass sim truth.

The value model never selects anything in this experiment (expand_select.py decides from the simulator alone), so
the discovery passes are a low-noise held-out test of V: the anchors come from a namespace no model was trained on,
and every row's truth is a 20-sample average instead of one 0/1 label.

Inputs (no simulator): the branch npz files (`latents[r:r+3]`, `agent_state_hist[r]` -- exactly the value model's
input at query round r), the discovery passes (truth: v_hat = mean v_target = mean gamma^T 1[0<=T<=B], T counted
from the QUERY state like V's training target, no anchor cap; p_hat = success rate) and discovery.json (class).
Scored rounds: the anchor at r=0 (its first branch, as the labeler does) and every branch at r = 2, 4, ..., R.

Metrics per model (value_model/<name>/model/value_ensemble.pt):
  rows      Spearman(V_R, v_hat_R), MAE, ECE (10 equal-width bins of V) over branches; Spearman(V_0, v_hat_0) over anchors
  ranking   per anchor: mean Spearman(V_R, v_hat_R) over its branches; best-of-N by V_R vs random vs oracle (v_hat_R)
  improved  AUROC of dV = V_R(branch) - V_0(anchor) for discovery class in {rescue, more_reliable, faster}
  early     Spearman(V_2, v_hat_R) and AUROC of V_2 - V_0 for the same label (SAILOR decides at r=2)
-> <out>/v_eval.json, v_eval.md, v_scores.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402
from expand_select import IMPROVED, branch_path  # noqa: E402
from value_model import ValueEnsemble  # noqa: E402


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(v: np.ndarray, t: np.ndarray, bins: int = 10) -> float:
    idx = np.clip((v * bins).astype(int), 0, bins - 1)
    return float(
        sum(
            abs(v[idx == b].mean() - t[idx == b].mean()) * (idx == b).mean()
            for b in range(bins)
            if (idx == b).any()
        )
    )


def sp(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def truth(files: list[Path]) -> tuple[int, dict, dict]:
    """(R, anchor seed -> (first branch, [v_target]), branch -> [v_target], [success])."""
    anc, br, R = {}, {}, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        R = int(max(d["r"]))
        for i in range(len(d["y"])):
            ok = bool(d["y"][i] | d["pre_success"][i])
            name, v = str(d["branch"][i]), float(d["v_target"][i])
            if int(d["r"][i]) == 0:
                rec = anc.setdefault(int(d["seed_id"][i]), {"branch": name, "v": [], "s": []})
            else:
                rec = br.setdefault(name, {"seed": int(d["seed_id"][i]), "v": [], "s": []})
            rec["v"].append(v)
            rec["s"].append(ok)
    return R, anc, br


@torch.no_grad()
def score(ve: ValueEnsemble, lat: np.ndarray, st: np.ndarray, device: str, bs: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(lat), bs):
        v, _r, _p = ve._forward_all(
            torch.as_tensor(lat[i : i + bs], dtype=torch.float32, device=device),
            torch.as_tensor(st[i : i + bs], dtype=torch.float32, device=device),
        )
        out.append(torch.sigmoid(v).mean(0).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", type=Path, default=None, help="stage-2 root (default <out_dir>/stage2)")
    ap.add_argument("--glob", default="labels_v*.npz")
    ap.add_argument("--discovery", type=Path, default=None, help="default <out_dir>/discovery.json")
    ap.add_argument("--out", type=Path, default=None, help="default <out_dir>")
    a = ap.parse_args()
    cfg = load_config(a.config, out_dir_copy=False)
    base = Path(cfg["out_dir"]).expanduser() / ("smoke" if a.smoke else "")
    root, out = a.root or base / "stage2", a.out or base
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R, anc, br = truth(sorted(root.glob(a.glob)))
    disc = {
        r["branch"]: r["cls"]
        for r in json.loads((a.discovery or base / "discovery.json").read_text())["rows"]
    }
    names = sorted(br)
    seeds = sorted(anc)
    rounds = list(range(2, R + 1, 2))

    def load(name, r):
        z = np.load(branch_path(root, name))
        return z["latents"][r : r + 3].astype(np.float32), z["agent_state_hist"][r].astype(np.float32)

    lat_a, st_a = map(np.stack, zip(*(load(anc[s]["branch"], 0) for s in seeds), strict=True))
    per_r = {r: tuple(map(np.stack, zip(*(load(n, r) for n in names), strict=True))) for r in rounds}
    vhat_b = np.array([np.mean(br[n]["v"]) for n in names])
    vhat_a = np.array([np.mean(anc[s]["v"]) for s in seeds])
    seed_of = np.array([br[n]["seed"] for n in names])
    a_idx = {s: i for i, s in enumerate(seeds)}
    improved = np.array([disc.get(n) in IMPROVED for n in names])
    report, scores = (
        {
            "R": R,
            "n_branches": len(names),
            "n_anchors": len(seeds),
            "n_improved": int(improved.sum()),
            "passes": len(next(iter(br.values()))["v"]),
            "models": {},
        },
        {"branch": np.array(names)},
    )
    for name in cfg["expand"]["v_models"]:
        path = Path(cfg["out_dir"]).expanduser().parent / name / "model" / "value_ensemble.pt"
        ve = ValueEnsemble.load(path, device=device)
        ve.eval()
        v0 = score(ve, lat_a, st_a, device)
        vr = {r: score(ve, *per_r[r], device) for r in rounds}
        vR, v0b = vr[R], v0[[a_idx[s] for s in seed_of]]
        within, regret = [], {"by_v": [], "random": [], "oracle": []}
        for s in seeds:
            m = seed_of == s
            if m.sum() < 2:
                continue
            within.append(sp(vR[m], vhat_b[m]))
            t = vhat_b[m]
            regret["by_v"].append(float(t.max() - t[np.argmax(vR[m])]))
            regret["random"].append(float(t.max() - t.mean()))
            regret["oracle"].append(0.0)
        report["models"][name] = {
            "rows": {
                "spearman_branch": sp(vR, vhat_b),
                "mae_branch": float(np.abs(vR - vhat_b).mean()),
                "ece_branch": ece(vR, vhat_b),
                "spearman_anchor": sp(v0, vhat_a),
                "mean_v": float(vR.mean()),
                "mean_vhat": float(vhat_b.mean()),
            },
            "ranking": {
                "mean_within_anchor_spearman": float(np.nanmean(within)) if within else float("nan"),
                "best_of_n_regret_by_v": float(np.mean(regret["by_v"])),
                "best_of_n_regret_random": float(np.mean(regret["random"])),
                "n_anchors": len(within),
            },
            "improved": {"auroc_dV": auroc(improved, vR - v0b), "auroc_V_R": auroc(improved, vR)},
            "early": {"spearman_V2_vs_vhatR": sp(vr[2], vhat_b), "auroc_dV2": auroc(improved, vr[2] - v0b)},
        }
        for r in rounds:
            scores[f"{name}_r{r}"] = vr[r]
        scores[f"{name}_anchor_r0"] = v0b
        print(name, json.dumps(report["models"][name]))
    scores.update(vhat=vhat_b, improved=improved, seed=seed_of)
    np.savez(out / "v_scores.npz", **scores)
    (out / "v_eval.json").write_text(json.dumps(report, indent=1, default=float))
    lines = [
        f"# V evaluation on the expand discovery passes (R={R}, {len(names)} branches, {len(seeds)} anchors, "
        f"{int(improved.sum())} improved)",
        "",
        "| model | Spearman V_R vs v̂ | MAE | ECE | Spearman V_0 (anchors) | within-anchor Spearman | "
        "regret by V / random | AUROC ΔV (improved) | AUROC ΔV at r=2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        lines.append(
            f"| {name} | {m['rows']['spearman_branch']:.3f} | {m['rows']['mae_branch']:.3f} | "
            f"{m['rows']['ece_branch']:.3f} | {m['rows']['spearman_anchor']:.3f} | "
            f"{m['ranking']['mean_within_anchor_spearman']:.3f} | {m['ranking']['best_of_n_regret_by_v']:.3f} / "
            f"{m['ranking']['best_of_n_regret_random']:.3f} | {m['improved']['auroc_dV']:.3f} | "
            f"{m['early']['auroc_dV2']:.3f} |"
        )
    (out / "v_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
