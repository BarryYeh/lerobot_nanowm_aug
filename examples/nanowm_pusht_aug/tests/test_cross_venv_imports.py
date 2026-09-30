"""Cross-venv import guard: research modules must load without lerobot/gym_pusht."""

from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import NANOWM_PY, PROJ, REPO

CROSS_VENV_MODULES = [
    "datagen.action_convert",
    "datagen.coverage_from_frame",
    "critic.value_model",
]


def _blocked_import(py: str, mod: str) -> subprocess.CompletedProcess[str]:
    code = (
        "import sys; "
        "sys.modules['lerobot']=None; "
        "sys.modules['gym_pusht']=None; "
        f"sys.path.insert(0, {str(PROJ)!r}); "
        f"import {mod}"
    )
    return subprocess.run(
        [py, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("mod", CROSS_VENV_MODULES)
def test_import_blocked_lerobot_venv(mod: str) -> None:
    r = _blocked_import(sys.executable, mod)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("mod", CROSS_VENV_MODULES)
def test_import_nanowm_venv(mod: str) -> None:
    if not NANOWM_PY.exists():
        pytest.skip("NanoWM venv missing")
    r = _blocked_import(str(NANOWM_PY), mod)
    assert r.returncode == 0, r.stderr


def _action_roundtrip(py: str) -> None:
    code = f"""
import sys
sys.path.insert(0, {str(PROJ)!r})
import numpy as np
from datagen import action_convert as ac
rng = np.random.default_rng(0)
pos0 = np.array([256.0, 300.0])
vel0 = np.zeros(2)
abs_t = pos0 + rng.normal(0, 40, size=(20, 2))
rel, agent_pos, _ = ac.abs_targets_to_rel(abs_t, pos0, vel0)
back = ac.rel_to_abs_targets(rel, agent_pos)
assert np.abs(back - abs_t).max() < 1e-9
print('ok')
"""
    r = subprocess.run([py, "-c", code], cwd=str(REPO), capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def _coverage_synth(py: str) -> None:
    code = f"""
import sys
sys.path.insert(0, {str(PROJ)!r})
import numpy as np
from datagen.coverage_from_frame import coverage_from_frame, T_FILL_RGB, goal_mask
blank = np.zeros((224, 224, 3), np.uint8)
d = coverage_from_frame(blank)
assert d['coverage'] == 0.0
frame = blank.copy()
gm = goal_mask(224)
frame[gm.astype(bool)] = T_FILL_RGB
d2 = coverage_from_frame(frame)
assert d2['coverage'] > 0.9
print('ok')
"""
    r = subprocess.run([py, "-c", code], cwd=str(REPO), capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_functional_lerobot_venv() -> None:
    _action_roundtrip(sys.executable)
    _coverage_synth(sys.executable)


def test_functional_nanowm_venv() -> None:
    if not NANOWM_PY.exists():
        pytest.skip("NanoWM venv missing")
    _action_roundtrip(str(NANOWM_PY))
    _coverage_synth(str(NANOWM_PY))
