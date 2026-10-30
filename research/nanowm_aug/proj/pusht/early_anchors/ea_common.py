"""Shared pieces of the early_anchors scripts: constants, rollout / source I/O, the anchor replay, pair datasets and
the paired statistics of the reports.

proj/ is not a package: the scripts import this as a sibling module (Python puts a script's own directory on
sys.path). Importing it also puts the import root research/nanowm_aug on sys.path, so `nwaug.*` and
`nwaug_envs.pusht.*` import after it. Heavy imports (gym_pusht, lerobot) stay inside the functions, so the report
scripts run without `--extra training`.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[3]  # research/nanowm_aug
if str(RES) not in sys.path:
    sys.path.insert(0, str(RES))

from nwaug.attribution.paired_report import (  # noqa: E402, F401
    boot_idx,
    mcnemar_exact as mcnemar,  # noqa: E402, F401
    pooled_diff,
)

CFG = RES / "configs" / "early_anchors.yaml"
FI, H, NF = 5, 3, 4  # configs/common.yaml nanowm: frame interval, history, num_frames
LEAD = (H - 1) * FI  # real actions a[t-10 : t] inside the first NanoWM window
EPISODE_STEPS = 300  # pusht.episode_length
EP_LEN = (
    9  # pair episode length, trained with drop_n_last_frames 8 (as last-mile / method1 / imagined_replay)
)
REAL_REPO = "local/pusht_imgonly"
ROLLOUT_KEYS = ("src_idx", "cand", "cov", "exec_acts")


# ------------------------------------------------------------------------------------------------------ simulator
def make_env(obs_type: str, size: int, max_steps: int):
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    kw = {"observation_width": size, "observation_height": size} if size else {}
    return gym.make("gym_pusht/PushT-v0", obs_type=obs_type, max_episode_steps=max_steps, **kw)


def load_source(source: Path, keys=("env_seed", "acts", "anchor_t")) -> dict:
    """Arrays of an `anchors.py source` run (<source>/episodes.npz), read into memory once."""
    z = np.load(source / "episodes.npz")
    return {k: z[k] for k in keys}


def replay_to_anchor(env, src: dict, i: int):
    """Exact anchor state of source episode i: env.reset(env_seed) + a[0 : anchor_t]. Returns the last (obs, info)."""
    o, _ = env.reset(seed=int(src["env_seed"][i]))
    info = {"coverage": 0.0}
    acts = src["acts"][i]
    for s in range(int(src["anchor_t"][i])):
        o, _, _, _, info = env.step(acts[s])
    return o, info


# ---------------------------------------------------------------------------------------------------- part files
def part_files(d: Path) -> list[Path]:
    """<d>/parts/part<i>.npz in part order (unfinished part<i>.tmp.npz files are skipped)."""
    files = [f for f in (d / "parts").glob("part*.npz") if f.stem[4:].isdigit()]
    return sorted(files, key=lambda f: int(f.stem[4:]))


def save_part(dest: Path, **arrays) -> None:
    """Atomic np.savez: a finished part is never half-written, so the resumable loops can skip it."""
    tmp = dest.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(dest)


def concat_parts(files: list[Path], keys=None) -> dict:
    zs = [np.load(f) for f in files]
    return {k: np.concatenate([z[k] for z in zs]) for k in keys or zs[0].files}


def load_rollout(rollout: Path, keys=ROLLOUT_KEYS) -> tuple[dict, dict, list[Path]]:
    """A `rollout.py run` output: its provenance, the branch arrays concatenated over parts plus `part` / `row` (where
    each branch lives, to fetch its frames later), and the part files."""
    prov = json.loads((rollout / "provenance.json").read_text())
    parts = part_files(rollout)
    zs = [np.load(f) for f in parts]
    d = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    d["part"] = np.concatenate([np.full(len(z["cov"]), i) for i, z in enumerate(zs)])
    d["row"] = np.concatenate([np.arange(len(z["cov"])) for z in zs])
    return prov, d, parts


def first_hit(cov: np.ndarray, thr: float) -> np.ndarray:
    """First round (0-based) with cov >= thr per row, -1 if never."""
    hit = cov >= thr
    return np.where(hit.any(1), hit.argmax(1), -1)


# ---------------------------------------------------------------------------------------------------- pair repos
def n_episodes(repo: str) -> int:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return int(json.loads((HF_LEROBOT_HOME / repo / "meta/info.json").read_text())["total_episodes"])


def real_sampleable() -> int:
    """Sampleable pairs of the real train set at drop_n_last_frames EP_LEN - 1."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    info = json.loads((HF_LEROBOT_HOME / REAL_REPO / "meta/info.json").read_text())
    return int(info["total_frames"] - (EP_LEN - 1) * info["total_episodes"])


def create_pair_repo(repo: str, overwrite: bool):
    from nwaug.datagen import pack_closed_loop

    return pack_closed_loop.create_pair_repo(repo, overwrite)


def add_pair(ds, frame96: np.ndarray, acts: np.ndarray) -> None:
    """One ep_len-9 pair episode (`pack_closed_loop.add_pair`), trained with drop_n_last_frames 8."""
    from nwaug.datagen import pack_closed_loop

    assert len(acts) == EP_LEN, len(acts)
    pack_closed_loop.add_pair(ds, frame96, acts)


def build_mix(pool: str, repo: str, rep: int) -> None:
    """<repo> = REAL_REPO + <pool> repeated rep times, with the PINNED stats reset to the real data's."""
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from nwaug.attribution.build_selected_datasets import PINNED

    home = HF_LEROBOT_HOME
    if (home / repo).exists():
        shutil.rmtree(home / repo)
    aggregate_datasets(
        repo_ids=[REAL_REPO] + [pool] * rep,
        aggr_repo_id=repo,
        roots=[home / REAL_REPO] + [home / pool] * rep,
        aggr_root=home / repo,
    )
    real = json.loads((home / REAL_REPO / "meta/stats.json").read_text())
    sp = home / repo / "meta/stats.json"
    st = json.loads(sp.read_text())
    for key in PINNED:
        st[key] = real[key]
    sp.write_text(json.dumps(st))


# ---------------------------------------------------------------------------------------------------- statistics
def boot_diff(dd: np.ndarray, idx: np.ndarray) -> dict:
    """Mean of a per-unit difference and its paired-bootstrap 95% CI."""
    lo, hi = np.percentile(dd[idx].mean(1), [2.5, 97.5])
    return {"mean": float(dd.mean()), "ci95": [float(lo), float(hi)]}


def restricted_steps(hit: np.ndarray, cap) -> np.ndarray:
    """Steps to the threshold, never (-1) and beyond-the-cap counted as the cap (restricted mean when averaged)."""
    return np.where(hit >= 0, np.minimum(hit, cap), cap)
