"""Shared pieces of the imagined_replay scripts: constants and seeds, the near-miss starts, and the last-mile W
anchors of the Img-W experiment.

proj/ is not a package: the scripts import this as a sibling module (Python puts a script's own directory on
sys.path). Importing it also puts the import root research/nanowm_aug on sys.path, so `nwaug.*` and
`nwaug_envs.pusht.*` import after it. gym_pusht stays inside the functions, so the offline steps skip it.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[3]  # research/nanowm_aug
if str(RES) not in sys.path:
    sys.path.insert(0, str(RES))

from nwaug.attribution.config_io import COMMON, load_config  # noqa: E402

_CFG = load_config(COMMON, out_dir_copy=False)
_NS = _CFG["seed_namespace"]

BASE_CKPT = _CFG["ckpt"]["base"]
FI = 5  # sim steps per NanoWM frame (= nwaug_envs.pusht.deploy.FI)
H, NF = int(_CFG["nanowm"]["history"]), int(_CFG["nanowm"]["num_frames"])
CTX = H * FI  # 15: F(s-15) is the first context frame, the action window starts there
NEED = NF * FI  # 20 actions: a[s-15 : s+5]
NSS, WM_BATCH = int(_CFG["nanowm"]["num_sampling_steps"]), int(_CFG["nanowm"]["wm_batch"])
NA = 10  # eval protocol n_action_steps
EP_LEN = 9  # pair episode length, trained with drop_n_last_frames 8 (as last-mile / method1; ctrl is reused)
REAL_REPO = "local/pusht_imgonly"

EVAL_SEED = int(_NS["eval"])  # eval_success_steps.py --seed: episode i uses env seed EVAL_SEED + i
SOURCE_SEED = int(_NS["imagined_replay_source"])
BRANCH_SEED = int(_NS["imagined_replay_branch"])
WM_SEED = int(_NS["imagined_replay_wm"])
SELECT_SEED = int(_NS["imagined_replay_select"])

LM = Path.home() / "datastorage/nanowm_pusht_aug/lastmile"  # heuristic/run_lastmile.sh output
LM_SIGMAS = ("0.0", "0.1", "0.25")
RESET_BASE = int(_NS["closed_loop_leadin"])  # closed_loop_rollout.run_lead_in: env.reset(seed=RESET_BASE + i)


# ---------------------------------------------------------------------------------------------- near-miss starts
def load_starts(path: Path) -> dict:
    """`collect_success.py starts` output (<collect>/starts.npz); prefix row i is valid for [:t_nm[i]]."""
    d = np.load(path)
    return {
        "env_seed": d["env_seed"],
        "t_nm": d["t_nm"],
        "cog": d["cog_at_t_nm"],
        "prefix": d["prefix_actions"],
    }


def start(starts: dict, i: int) -> dict:
    """Start i: episode seed, t_nm, block CoG at t_nm and the executed actions a[0 : t_nm]."""
    t_nm = int(starts["t_nm"][i])
    return {
        "env_seed": int(starts["env_seed"][i]),
        "t_nm": t_nm,
        "cog": starts["cog"][i],
        "prefix": starts["prefix"][i, :t_nm],
    }


# ------------------------------------------------------------------------------------------- last-mile W anchors
def w_anchors(lm: Path = LM):
    """The last-mile W pairs in file order: yields (file, branch npz, sigma, lead-in seed index i, lead-in npz)."""
    lead = {s: np.load(lm / "imag" / f"sigma{s}" / "work" / "lead_in.npz") for s in LM_SIGMAS}
    for f in sorted((lm / "sel" / "W").glob("*.npz")):
        z = np.load(f)
        sig = f.name.split("_")[0].removeprefix("sigma")
        yield f, z, sig, int(z["seed_index"]), lead[sig]


def lastmile_env(i: int, seed_state):
    """224px pixel env at lead-in step 0 of last-mile seed i: env.reset(RESET_BASE + i), then the saved body state."""
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    from nwaug_envs.pusht.state import place_body

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=224, observation_height=224)
    env.reset(seed=RESET_BASE + i)
    place_body(env.unwrapped, seed_state)
    return env


def sha1(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes(), usedforsecurity=False).hexdigest()
