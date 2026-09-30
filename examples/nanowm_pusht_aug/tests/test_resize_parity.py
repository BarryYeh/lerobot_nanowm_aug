"""closed_loop_rollout._to96 and pack_closed_loop.to96 must match (M3)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gym_pusht")

from closed_loop_rollout import _to96  # noqa: E402
from datagen.pack_closed_loop import to96  # noqa: E402


def test_to96_parity_on_random_224_frames() -> None:
    rng = np.random.default_rng(0)
    for _ in range(16):
        f = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        a = _to96(f)
        b = to96(f)
        assert a.dtype == np.uint8 and b.dtype == np.uint8
        assert a.shape == (96, 96, 3)
        assert np.array_equal(a, b)
