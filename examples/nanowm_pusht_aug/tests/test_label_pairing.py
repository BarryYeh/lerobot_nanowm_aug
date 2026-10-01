"""`label_imagined_branches` cross-branch asserts: shared lead-in per anchor, paired round-0 chunks per cand."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gym_pusht")

from label_imagined_branches import (  # noqa: E402
    PAIRED_CHUNK_TOL_PX,
    assert_paired_candidate,
    assert_shared_lead_in,
)


def _branch(name: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "name": name,
        "lead_acts": rng.uniform(100, 400, (10, 2)),
        "anchor_state": rng.uniform(100, 400, 5),
        "agent_state_hist": rng.uniform(-5, 5, (9, 4)),
        "chunk0_clean": rng.uniform(100, 400, (16, 2)),
    }


def test_shared_lead_in_accepts_identical_and_rejects_any_difference():
    a, b = _branch("sigma0.0/seed0000_cand0.npz"), _branch("sigma0.1/seed0000_cand1.npz")
    assert_shared_lead_in(a, b, b["name"])
    for key, idx in (("lead_acts", (3, 1)), ("anchor_state", (2,)), ("agent_state_hist", (0, 2))):
        c = {**b, key: b[key].copy()}
        c[key][idx] += 1e-12
        with pytest.raises(AssertionError, match=key.split("_hist")[0]):
            assert_shared_lead_in(a, c, c["name"])
    later = {**b, "agent_state_hist": b["agent_state_hist"].copy()}
    later["agent_state_hist"][1:] += 3.0  # rounds after the anchor may differ (different chunks / noise)
    assert_shared_lead_in(a, later, later["name"])


def test_paired_candidate_tolerance():
    a, b = _branch("sigma0.0/seed0000_cand0.npz"), _branch("sigma0.1/seed0000_cand0.npz")
    assert_paired_candidate(a, b, b["name"])
    c = {**b, "chunk0_clean": b["chunk0_clean"] + 0.5 * PAIRED_CHUNK_TOL_PX}
    assert_paired_candidate(a, c, c["name"])
    d = {**b, "chunk0_clean": b["chunk0_clean"].copy()}
    d["chunk0_clean"][7, 0] += 2 * PAIRED_CHUNK_TOL_PX
    with pytest.raises(AssertionError, match="chunk0_clean"):
        assert_paired_candidate(a, d, d["name"])
