"""Research-code test scaffolding (stdlib only at import time)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parents[1]
NANOWM_PY = REPO / "third_party" / "nano-world-model" / ".venv" / "bin" / "python"

# One import root (PROJ) for `nwaug` / `nwaug_envs`. The experiment folders are not packages:
# tests add them so scripts can import their same-folder siblings, as they do when run directly.
EXPERIMENTS = [
    PROJ / "proj" / "pusht" / n
    for n in ("sailor", "cupid", "heuristic", "method1", "imagined_replay")
]
for p in (PROJ, *EXPERIMENTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
