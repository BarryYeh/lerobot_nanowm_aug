"""Helpers from imagined_rollout that must work in the NanoWM venv without a dataset/GPU."""

from __future__ import annotations

import subprocess

import pytest

from conftest import NANOWM_PY, PROJ, REPO


def test_seq_length_nanowm_venv() -> None:
    if not NANOWM_PY.exists():
        pytest.skip("NanoWM venv missing")
    code = f"""
import sys
sys.path.insert(0, {str(PROJ)!r})
import imagined_rollout as ir
from types import SimpleNamespace

# import alone must not need dataset / GPU / ckpt
assert hasattr(ir, "_seq_length")

with_method = SimpleNamespace(get_seq_length=lambda i: 7)
assert ir._seq_length(with_method, 0) == 7
assert type(ir._seq_length(with_method, 0)) is int

with_attr = SimpleNamespace(seq_lengths=[3, 11])
assert ir._seq_length(with_attr, 1) == 11
assert type(ir._seq_length(with_attr, 1)) is int
print("ok")
"""
    r = subprocess.run(
        [str(NANOWM_PY), "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
