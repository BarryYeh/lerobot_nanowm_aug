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


def test_closed_loop_seed_default_and_semantics() -> None:
    """`closed-loop --seed` defaults to None (unseeded worker, the original behaviour); `seed_wm(None, .)`
    must not touch the RNG, `seed_wm(s, o)` must equal `torch.manual_seed(s + o)`."""
    if not NANOWM_PY.exists():
        pytest.skip("NanoWM venv missing")
    code = f"""
import sys
sys.path.insert(0, {str(PROJ)!r})
import torch
import imagined_rollout as ir

p = ir.build_parser()
assert p.parse_args(["closed-loop", "--work", "w"]).seed is None
assert p.parse_args(["closed-loop", "--work", "w", "--seed", "7"]).seed == 7
assert p.parse_args(["recon", "--in", "a.npz", "--out", "b.npz"]).seed == 0

torch.manual_seed(123)
state = torch.get_rng_state()
ir.seed_wm(None, 3)
assert torch.equal(torch.get_rng_state(), state), "seed_wm(None) changed the RNG"

ir.seed_wm(5, 1)
a = torch.randn(4)
torch.manual_seed(6)
assert torch.equal(a, torch.randn(4))
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
