"""Git / argv provenance helpers (stdlib only; both venvs)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_info(repo_root: Path | None = None) -> dict:
    """Return ``{git_sha, git_dirty, git_branch}``; all None on any failure."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    git = shutil.which("git")
    out = {"git_sha": None, "git_dirty": None, "git_branch": None}
    if not git:
        return out
    try:
        sha = subprocess.check_output(
            [git, "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = (
            subprocess.check_output(
                [git, "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            != ""
        )
        branch = subprocess.check_output(
            [git, "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"git_sha": sha, "git_dirty": dirty, "git_branch": branch}
    except Exception:  # noqa: BLE001 -- never raise from provenance
        return out


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def provenance_entry(args, argv=None, **extra) -> dict:
    """Build one provenance dict: created_utc, argv, args, git_info, extras."""
    if argv is None:
        argv = list(sys.argv)
    if args is None:
        args_dict = None
    elif hasattr(args, "__dict__"):
        args_dict = _json_safe(vars(args))
    else:
        args_dict = _json_safe(args)
    entry = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": list(argv),
        "args": args_dict,
        **git_info(),
        **extra,
    }
    # ensure json-serializable
    json.dumps(entry)
    return entry
