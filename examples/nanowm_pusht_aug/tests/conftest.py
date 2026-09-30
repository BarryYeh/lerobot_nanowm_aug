"""Research-code test scaffolding (stdlib only at import time)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parents[1]
NANOWM_PY = REPO / "third_party" / "nano-world-model" / ".venv" / "bin" / "python"

for p in (PROJ, PROJ / "eval", PROJ / "attribution", PROJ / "critic"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
