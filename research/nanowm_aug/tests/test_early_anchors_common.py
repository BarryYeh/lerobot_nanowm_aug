"""`proj/pusht/early_anchors/ea_common.py`: rollout part I/O and the paired statistics of the reports."""

from __future__ import annotations

import numpy as np
from ea_common import first_hit, load_rollout, pooled_diff, restricted_steps, save_part


def test_load_rollout_concatenates_parts_and_skips_unfinished(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "provenance.json").write_text('{"k": 2}')
    for p, n in ((0, 2), (1, 3), (10, 1)):
        save_part(
            tmp_path / "parts" / f"part{p}.npz", cov=np.full((n, 4), p, np.float32), src_idx=np.full(n, p)
        )
    np.savez(tmp_path / "parts" / "part2.tmp.npz", cov=np.zeros((5, 4)), src_idx=np.zeros(5))
    prov, d, parts = load_rollout(tmp_path, keys=("cov", "src_idx"))
    assert prov == {"k": 2} and [f.name for f in parts] == ["part0.npz", "part1.npz", "part10.npz"]
    assert d["src_idx"].tolist() == [0, 0, 1, 1, 1, 10]
    assert d["part"].tolist() == [0, 0, 1, 1, 1, 2] and d["row"].tolist() == [0, 1, 0, 1, 2, 0]


def test_first_hit_and_restricted_steps():
    cov = np.array([[0.1, 0.95, 0.99], [0.1, 0.2, 0.3]])
    assert first_hit(cov, 0.9).tolist() == [1, -1]
    assert restricted_steps(np.array([[5, -1], [400, 10]]), np.array([[300], [200]])).tolist() == [
        [5, 300],
        [200, 10],
    ]


def test_pooled_diff_is_the_mean_over_seeds():
    x = [np.array([1, 1, 0, 0], bool), np.array([1, 1, 1, 0], bool)]
    y = [np.array([1, 0, 0, 0], bool)] * 2
    idx = np.random.default_rng(0).integers(0, 4, size=(200, 4))
    r = pooled_diff(x, y, idx)
    assert r["per_seed"] == [0.25, 0.5] and r["pooled"] == 0.375
    assert r["ci95"][0] <= r["pooled"] <= r["ci95"][1]
