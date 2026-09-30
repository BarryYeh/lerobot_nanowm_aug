"""Tests for datagen.provenance."""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

from datagen.provenance import git_info, provenance_entry

from conftest import REPO


def test_git_info_in_repo() -> None:
    info = git_info(REPO)
    assert info["git_sha"] is not None
    assert len(info["git_sha"]) == 40
    assert all(c in "0123456789abcdef" for c in info["git_sha"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    assert info["git_sha"] == head
    assert isinstance(info["git_dirty"], bool)
    assert isinstance(info["git_branch"], str)


def test_git_info_non_repo(tmp_path: Path) -> None:
    info = git_info(tmp_path)
    assert info == {"git_sha": None, "git_dirty": None, "git_branch": None}


def test_provenance_entry_json_safe() -> None:
    args = Namespace(ep_len=9, root=Path("/tmp/x"))
    entry = provenance_entry(args, mode="pack", n_pairs=3)
    json.dumps(entry)
    assert entry["args"]["ep_len"] == 9
    assert entry["args"]["root"] == "/tmp/x"
    assert entry["mode"] == "pack"
    assert "created_utc" in entry
