#!/usr/bin/env python
"""Keep the first N episodes of an existing evaluation (eval_info.json + success_steps.npz).

Episode i of an `eval_success_steps.py --seed 100000` run uses env seed 100000 + i whatever --eval.n_episodes is, so
the first N episodes of a 2000-episode evaluation are the same episodes as an N-episode evaluation.

    uv run --extra training python .../truncate_eval.py <src eval dir> <dest dir> <N>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ir_common import EVAL_SEED


def main() -> None:
    src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    info = json.loads((src / "eval_info.json").read_text())
    m = info["per_task"][0]["metrics"]
    assert len(m["successes"]) >= n, (src, len(m["successes"]), n)
    for k in ("sum_rewards", "max_rewards", "successes"):
        m[k] = m[k][:n]
    info["truncated_from"] = str(src)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "eval_info.json").write_text(json.dumps(info))
    z = np.load(src / "success_steps.npz")
    assert np.array_equal(z["seed"][:n], EVAL_SEED + np.arange(n)), src
    np.savez(dst / "success_steps.npz", **{k: z[k][:n] for k in z.files})
    print(f"{dst.name}: first {n} episodes of {src}, success {np.mean(m['successes']):.4f}")


if __name__ == "__main__":
    main()
